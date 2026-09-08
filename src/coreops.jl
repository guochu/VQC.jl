# coreops.jl — 通用态操作原语（核心层，不依赖任何 IR 包）
#
# 本文件提供与"局域矩阵 / Kraus 算子集 + 比特位置"打交道的原语；
# 对 QuantumCircuits 指令（GateOp / ChannelOp / …）的适配见
# `ext/VQCQuantumCircuitsExt.jl`。

# 约定：
#   * 比特位置 1-based、小端序（qubit 1 = 最低有效位，与 QuantumCircuits 一致）；
#   * `positions[1]` 为矩阵最高位（与 QuantumCircuits 的门矩阵约定一致）；
#   * 内部位运算（kernel key 等）仍为 0-based。

"""
位置元组（MSB-first，1-based）→ kernel key（LSB-first，0-based 位号）。内部工具。
"""
_lsb_key(positions::Vector{Int}) = ntuple(i -> positions[length(positions)+1-i] - 1, Val(length(positions)))

_ensure_complex(s::StateVector) =
    eltype(s) <: Complex ? s :
    StateVector(convert(Vector{complex(float(eltype(s)))}, storage(s)), _nqubits(s))
_ensure_complex(s::DensityMatrix) =
    eltype(s) <: Complex ? s :
    DensityMatrix(convert(Vector{complex(float(eltype(s)))}, s.data), _nqubits(s))

function _check_positions(positions::Vector{Int}, n::Int)
    length(unique(positions)) == length(positions) || throw(ArgumentError("duplicate qubit positions"))
    all(q -> 1 <= q <= n, positions) || throw(ArgumentError("qubit position out of range [1, $n]"))
    return positions
end

"""
    apply(state, m::AbstractMatrix, positions::Vector{Int}) -> state

把 `length(positions)` 比特局域矩阵 `m`（`positions[1]` = 矩阵最高位，
1-based 比特位置）作用到态上：

* `StateVector`：`ψ ← m ψ`；
* `DensityMatrix`：`ρ ← m ρ m†`。

实数态遇复矩阵自动提升为 `ComplexF64`（返回值可能是新对象）。
"""
function apply(s::StateVector, m::AbstractMatrix, positions::Vector{Int})
    _check_positions(positions, _nqubits(s))
    size(m, 1) == size(m, 2) == 1 << length(positions) ||
        throw(ArgumentError("matrix size $(size(m)) does not match $(length(positions)) qubit(s)"))
    if eltype(s) <: Real && !(eltype(m) <: Real)
        s = _ensure_complex(s)
    end
    apply_kernel!(storage(s), _lsb_key(positions), m)
    return s
end

function apply(s::DensityMatrix, m::AbstractMatrix, positions::Vector{Int})
    n = _nqubits(s)
    _check_positions(positions, n)
    size(m, 1) == size(m, 2) == 1 << length(positions) ||
        throw(ArgumentError("matrix size $(size(m)) does not match $(length(positions)) qubit(s)"))
    if eltype(s) <: Real && !(eltype(m) <: Real)
        s = _ensure_complex(s)
    end
    key = _lsb_key(positions)
    apply_kernel!(s.data, key, m)                                        # ρ ← m ρ（行索引）
    apply_kernel!(s.data, ntuple(i -> key[i] + n, Val(length(positions))), conj(m))  # ρ ← ρ m†（列索引）
    return s
end

"""
    apply_kraus!(state, ks, positions) -> state

把 Kraus 算子集 `ks`（向量 `k` 满足 `Σₖ kₖ† kₖ = I`）作用到态上：

* `DensityMatrix`：`ρ ← Σₖ kₖ ρ kₖ†`（就地）；
* `StateVector`：先转换为 `DensityMatrix`（返回值类型因此改变）。

实现：先把 Kraus 集合组合成单个局域超算子
`m = Σₖ kron(kₖ, conj(kₖ))`（`D²×D²`），再经 `apply_kernel_dm!` 一次
作用到密度矩阵的平坦存储上——过程中不复制整个量子态。

算子矩阵转换到态的 `eltype` 后作用；实数态遇复算子自动提升为
复数版本（返回值可能是新对象）。
"""
function apply_kraus!(s::StateVector, ks, positions::Vector{Int})
    if eltype(s) <: Real && any(K -> !(eltype(K) <: Real), ks)
        s = _ensure_complex(s)
    end
    return apply_kraus!(DensityMatrix(s), ks, positions)
end

function apply_kraus!(s::DensityMatrix, ks, positions::Vector{Int})
    n = _nqubits(s)
    _check_positions(positions, n)
    isempty(ks) && return s
    D = size(ks[1], 1)
    D == 1 << length(positions) ||
        throw(ArgumentError("operator dimension $(D) does not match $(length(positions)) qubit(s)"))
    if eltype(s) <: Real && any(K -> !(eltype(K) <: Real), ks)
        s = _ensure_complex(s)
    end
    T = eltype(s)
    M = zeros(T, D * D, D * D)
    for K in ks
        size(K, 1) == size(K, 2) == D ||
            throw(ArgumentError("all Kraus operators must be $(D)×$(D)"))
        # ρ'[(i,j)] = Σₖ Σ_{i',j'} k[i,i'] ρ[i',j'] conj(k[j,j'])
        # 平坦布局 ρ[i + d·j] 中行索引为低位 → 超算子 = kron(conj(k), k)
        M .+= kron(conj(convert(Matrix{T}, K)), convert(Matrix{T}, K))
    end
    apply_kernel_dm!(s.data, 1 << n, _lsb_key(positions), M)
    return s
end

"""
    reset_qubit_zero!(state, q::Int) -> state

把 qubit `q` 相干重置到 `|0⟩`（Kraus：`k₀ = |0⟩⟨0|`、`k₁ = |0⟩⟨1|`）：

* `StateVector`：`ψ'ₓ₀ = ψₓ₀ + ψₓ₁` 后归一化；
* `DensityMatrix`：`ρ'₍ₓ₀,y₀₎ = ρ₍ₓ₀,y₀₎ + ρ₍ₓ₁,y₁₎` 后归一化。
"""
function reset_qubit_zero!(s::StateVector, q::Int)
    n = _nqubits(s)
    1 <= q <= n || throw(ArgumentError("qubit index $q out of range [1, $n]"))
    q -= 1
    v = storage(s)
    pos = 1 << q
    @inbounds for r in 0:(length(v) >> 1) - 1
        base = _bit_insert_zeros(r, (q,))
        i0 = base + 1
        v[i0] += v[i0 + pos]
        v[i0 + pos] = zero(eltype(v))
    end
    nrm = norm(v)
    nrm > 0 || throw(ArgumentError("cannot reset: resulting state is zero"))
    rmul!(v, 1 / nrm)
    return s
end

function reset_qubit_zero!(s::DensityMatrix, q::Int)
    n = _nqubits(s)
    1 <= q <= n || throw(ArgumentError("qubit index $q out of range [1, $n]"))
    q -= 1
    d = 1 << n
    data = s.data
    pos = 1 << q
    # ρ'[(x,0),(y,0)] = ρ[(x,0),(y,0)] + ρ[(x,1),(y,1)]
    @inbounds for r in 0:(d >> 1) - 1
        base = _bit_insert_zeros(r, (q,))
        i0 = base
        i1 = base + pos
        for c in 0:(d >> 1) - 1
            cbase = _bit_insert_zeros(c, (q,))
            j0 = cbase
            j1 = cbase + pos
            data[i0 + d * j0 + 1] += data[i1 + d * j1 + 1]
        end
    end
    # 清零行/列中 bit q = 1 的分支
    @inbounds for idx in 0:d-1
        if (idx >> q) & 1 == 1
            z = zero(eltype(data))
            for c in 0:d-1
                data[idx + d * c + 1] = z
            end
            for r in 0:d-1
                data[r + d * idx + 1] = z
            end
        end
    end
    tr = _dm_trace(data, d)
    tr > 0 || throw(ArgumentError("cannot reset: resulting state is zero"))
    rmul!(data, 1 / tr)
    return s
end

_dm_trace(data::AbstractVector, d::Int) = real(sum(data[i * d + i + 1] for i in 0:d-1))
