# hamiltonian.jl — 核心层：一般矩阵期望值、自旋算符代数与高效作用
#
# Pauli 代数（PauliTerm / PauliSum）的期望值见
# `ext/quantumcircuits/hamiltonian.jl`（QuantumCircuits 接口层）。

"""
    expectation(m::AbstractMatrix, s::Union{StateVector,DensityMatrix}) -> Complex

一般（可显式构造的）算符期望值：`⟨ψ|m|ψ⟩` 或 `tr(ρ m)`。
大矩阵直接走 BLAS；小局域算符建议用局域核（见 Pauli 接口层方法）。
"""
function expectation(m::AbstractMatrix, s::StateVector)
    size(m, 1) == size(m, 2) == length(s) || throw(DimensionMismatch("matrix size mismatch"))
    return dot(storage(s), m, storage(s))
end

function expectation(m::AbstractMatrix, s::DensityMatrix)
    d = 1 << _nqubits(s)
    size(m, 1) == size(m, 2) == d || throw(DimensionMismatch("matrix size mismatch"))
    return sum(vec(storage(s)) .* vec(transpose(m)))   # tr(ρ m) = Σ ρ_ij m_ji
end

"""
    expectation(state_c::StateVector, m::AbstractMatrix, state::StateVector) -> Complex

一般矩阵元 `⟨state_c|m|state⟩`（供自动微分与振幅放大使用）。
"""
function expectation(state_c::StateVector, m::AbstractMatrix, state::StateVector)
    length(state_c) == length(state) == size(m, 1) == size(m, 2) ||
        throw(DimensionMismatch("dimension mismatch"))
    return dot(storage(state_c), m, storage(state))
end

# ── 自旋算符代数（SpinOpTerm / SpinOpSum） ──────────────────────────────────
#
# PauliTerm / PauliSum 的核心层推广：单个比特位上可以放任意单比特算子
# （`Symbol` 简写或任意 `2^k × 2^k` 矩阵），不限于 Pauli 矩阵。
#
#   SpinOpTerm(0.5, 1 => :Z, 2 => :Y)          # 0.5 · Z₁ Y₂
#   SpinOpTerm(1.0, 1 => [0 1; -1 0])          # 任意 2×2 矩阵
#   SpinOpTerm(0.1, 3 => :X) + SpinOpTerm(1.0, 2 => :Z)² ∈ SpinOpSum
#
# 乘积结构使 `apply` 无需构造大矩阵：逐位走局域 kernel，
# 复杂度 O(#ops · 2ⁿ · D_local)，可直接用于高效本征态求解
# （Lanczos / DMRG 的作用算符）与时间演化（Trotter 步）。

const _SPIN1 = Dict{Symbol,Matrix{ComplexF64}}(
    :I => Matrix{ComplexF64}(I, 2, 2),
    :X => [0.0 1.0; 1.0 0.0],
    :Y => [0.0 -im; im 0.0],
    :Z => [1.0 0.0; 0.0 -1.0],
    :P => [0.0 1.0; 0.0 0.0],    # σ₊ = (X + iY) / 2
    :M => [0.0 0.0; 1.0 0.0],    # σ₋ = (X - iY) / 2
)

const _SPIN_ADJ = Dict(:I => :I, :X => :X, :Y => :Y, :Z => :Z, :P => :M, :M => :P)

"单比特算子：`Symbol` 简写或任意 `2×2` 矩阵（不限 Pauli / 厄米）。"
function _spin_matrix(op)
    if op isa Symbol
        haskey(_SPIN1, op) || throw(ArgumentError("unknown spin operator :$(op)"))
        return _SPIN1[op]
    elseif op isa AbstractMatrix
        size(op) == (2, 2) ||
            throw(ArgumentError("spin operator must be a 2×2 matrix, got $(size(op))"))
        return op
    end
    throw(ArgumentError("invalid spin operator $(typeof(op))"))
end

"""
    SpinOpTerm(coeff, ops...) -> SpinOpTerm

自旋算符乘积项：`coeff · A₁ A₂ …`，`Aᵢ` 作用在比特 `qᵢ` 上。
算子 `Aᵢ` 可以是 `Symbol` 简写（`:I / :X / :Y / :Z / :P / :M`）
或**任意 `2×2` 矩阵**（不限 Pauli / 厄米）。

* `SpinOpTerm(0.5, 1 => :Z, 2 => :Y)`
* `SpinOpTerm(1.0, 3 => [0 1; -1 0])`       # 任意 2×2 矩阵

位置自动排序；同一位重复出现按乘积顺序保留。
"""
struct SpinOpTerm
    coeff::ComplexF64
    ops::Vector{Pair{Int,Any}}
    function SpinOpTerm(coeff::Number, ops::Vector{<:Pair{Int}})
        isempty(ops) && throw(ArgumentError("SpinOpTerm requires at least one operator"))
        for p in ops
            _spin_matrix(last(p))          # 构造时校验算子合法性
        end
        new(ComplexF64(coeff), sort(ops; by = first))
    end
end
SpinOpTerm(coeff::Number, ops::Pair{Int}...) = SpinOpTerm(coeff, collect(ops))

Base.copy(t::SpinOpTerm) = SpinOpTerm(t.coeff, copy(t.ops))
Base.:(==)(t1::SpinOpTerm, t2::SpinOpTerm) = t1.coeff == t2.coeff && t1.ops == t2.ops
Base.hash(t::SpinOpTerm, h::UInt) = hash(t.ops, hash(t.coeff, h))

"""
    SpinOpSum(terms...) -> SpinOpSum

厄米（或一般）算符 = 自旋算符项之和。
"""
struct SpinOpSum
    terms::Vector{SpinOpTerm}
end
SpinOpSum(terms::SpinOpTerm...) = SpinOpSum(collect(terms))
SpinOpSum() = SpinOpSum(SpinOpTerm[])

Base.copy(s::SpinOpSum) = SpinOpSum(copy(s.terms))
Base.push!(s::SpinOpSum, t::SpinOpTerm) = (push!(s.terms, t); s)
Base.append!(s::SpinOpSum, t::SpinOpSum) = (append!(s.terms, t.terms); s)

Base.:+(t1::SpinOpTerm, t2::SpinOpTerm) = SpinOpSum([t1, t2])
Base.:+(s::SpinOpSum, t::SpinOpTerm) = push!(copy(s), t)
Base.:+(t::SpinOpTerm, s::SpinOpSum) = push!(copy(s), t)
Base.:+(s1::SpinOpSum, s2::SpinOpSum) = append!(copy(s1), s2)
Base.:-(t::SpinOpTerm) = SpinOpTerm(-t.coeff, t.ops)
Base.:*(c::Real, t::SpinOpTerm) = SpinOpTerm(c * t.coeff, t.ops)
Base.:*(t::SpinOpTerm, c::Real) = c * t
Base.:*(c::Real, s::SpinOpSum) = SpinOpSum([c * t for t in s.terms])
Base.:*(s::SpinOpSum, c::Real) = c * s

function Base.adjoint(t::SpinOpTerm)
    ops = Pair{Int,Any}[first(p) => (last(p) isa Symbol ? _SPIN_ADJ[last(p)] :
                                     adjoint(_spin_matrix(last(p)))) for p in t.ops]
    return SpinOpTerm(conj(t.coeff), ops)
end
Base.adjoint(s::SpinOpSum) = SpinOpSum([adjoint(t) for t in s.terms])
LinearAlgebra.ishermitian(t::SpinOpTerm) = t == adjoint(t)
LinearAlgebra.ishermitian(s::SpinOpSum) = s == adjoint(s)

"""
    mat(t::SpinOpTerm, n::Int) -> Matrix
    mat(s::SpinOpSum, n::Int) -> Matrix

展开为 `n` 比特的稠密矩阵（小端序；仅供小规模验证 / 对拍，
大规模使用请走 `apply` 的局域核路径）。
"""
function _embed_local(m::AbstractMatrix, key::NTuple{N,Int}, n::Int) where {N}
    d = 1 << n
    D = size(m, 1)
    full = zeros(ComplexF64, d, d)
    for col in 0:d-1
        lcol = 0
        ok = true
        for (j, b) in enumerate(key)
            lcol |= ((col >> b) & 1) << (j - 1)
        end
        for lrow in 0:D-1
            grow = col
            for (j, b) in enumerate(key)
                grow = (grow & ~(1 << b)) | (((lrow >> (j - 1)) & 1) << b)
            end
            full[grow+1, col+1] = m[lrow+1, lcol+1]
        end
    end
    return full
end

function mat(t::SpinOpTerm, n::Int)
    d = 1 << n
    acc = Matrix{ComplexF64}(I, d, d)          # 乘积项：按 ops 顺序右乘各因子的嵌入
    for (q, op) in t.ops
        m = _spin_matrix(op)
        1 <= q <= n || throw(ArgumentError("operator position $q out of range [1, $n]"))
        acc = acc * _embed_local(m, (q - 1,), n)
    end
    return t.coeff .* acc
end

function mat(s::SpinOpSum, n::Int)
    d = 1 << n
    acc = zeros(ComplexF64, d, d)
    for t in s.terms
        acc .+= mat(t, n)
    end
    return acc
end

# ── 高效作用：apply(SpinOpTerm, StateVector) ────────────────────────────────

"""
    apply(t::SpinOpTerm, s::StateVector) -> StateVector
    apply(s::SpinOpSum, s0::StateVector) -> StateVector

把自旋算符作用到态矢量上（**非就地**，返回新态）。

乘积项 `t = c · A₁ A₂ …` 无需构造大矩阵：逐因子走局域 kernel，
复杂度 `O(#ops · 2ⁿ · D_local)`。`SpinOpSum` 的作用为各项的线性组合。
适用于高效本征态求解（Lanczos / 精确对角化的作用算符）与
时间演化（Trotter 步：`exp(-i·Δt·H) ≈ Πₖ apply(termₖ, ·)`）。

实数态遇复算子自动提升为复数版本。
"""
function apply(t::SpinOpTerm, s::StateVector)
    v = copy(storage(s))
    n = _nqubits(s)
    for (q, op) in reverse(t.ops)      # 逆序作用 = 正序矩阵积 A₁ A₂ ⋯
        m = _spin_matrix(op)
        1 <= q <= n || throw(ArgumentError("operator position $q out of range [1, $n]"))
        if eltype(v) <: Real && !(eltype(m) <: Real)
            v = convert(Vector{complex(float(eltype(v)))}, v)
        end
        apply_kernel!(v, (q - 1,), m)
    end
    rmul!(v, t.coeff)
    return StateVector(v, _nqubits(s))
end

function apply(s::SpinOpSum, s0::StateVector)
    isempty(s.terms) && throw(ArgumentError("empty SpinOpSum"))
    out = copy(storage(apply(first(s.terms), s0)))
    for t in @view s.terms[2:end]
        out .+= storage(apply(t, s0))
    end
    return StateVector(out, _nqubits(s0))
end

"""
    expectation(t::SpinOpTerm, s::StateVector) -> Complex
    expectation(s::SpinOpSum, s::StateVector) -> Complex
    expectation(t::SpinOpTerm, s::DensityMatrix) -> Complex
    expectation(s::SpinOpSum, s::DensityMatrix) -> Complex

自旋算符期望值：态矢量走 `apply` 的局域核路径（`⟨ψ|t|ψ⟩`），
密度矩阵走局域 / 满矩阵展开（`tr(ρ t)`）。
"""
function expectation(t::SpinOpTerm, s::StateVector)
    return dot(storage(s), storage(apply(t, s)))
end

function expectation(s::SpinOpSum, st::StateVector)
    acc = zero(promote_type(ComplexF64, eltype(st)))
    for t in s.terms
        acc += dot(storage(st), storage(apply(t, st)))   # apply 已含 t.coeff
    end
    return acc
end

function expectation(t::SpinOpTerm, s::DensityMatrix)
    n = _nqubits(s)
    return expectation(mat(t, n), s)
end

function expectation(s::SpinOpSum, st::DensityMatrix)
    return expectation(mat(s, _nqubits(st)), st)   # mat 已含各项 coeff
end
