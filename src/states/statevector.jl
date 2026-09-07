# statevector.jl — 纯态（态矢量）
#
# 核心层：不依赖 QuantumCircuits 的任何类型。
# 内部协议 `_nqubits` 由接口层桥接到 QuantumCircuits 的 `nqubits`。

"""
纯态：长度 `2^n` 的振幅向量，小端序（qubit 0 = 最低有效位）。

    StateVector(data, n)          # 显式比特数
    StateVector(data)             # 自动推断（长度须为 2 的幂）
    StateVector(ComplexF64, n)    # |0…0⟩
    StateVector(n)
"""
struct StateVector{T<:Number}
    data::Vector{T}
    n::Int

    function StateVector{T}(data::AbstractVector{<:Number}, n::Integer) where {T<:Number}
        n >= 0 || throw(ArgumentError("number of qubits must be non-negative"))
        length(data) == 1 << n ||
            throw(DimensionMismatch("expected length $(1 << n) for $n qubits, got $(length(data))"))
        new{T}(convert(Vector{T}, data), Int(n))
    end
end

StateVector(data::AbstractVector{T}, n::Integer) where {T<:Number} = StateVector{T}(data, n)
function StateVector(data::AbstractVector{T}) where {T<:Number}
    L = length(data)
    (L >= 2 && L & (L - 1) == 0) || throw(ArgumentError("length must be a power of two, got $L"))
    return StateVector{T}(data, Int(log2(L)))
end
StateVector(data::AbstractVector{<:Number}, ::Nothing) = StateVector(data)
function StateVector{T}(n::Integer) where {T<:Number}
    n >= 0 || throw(ArgumentError("number of qubits must be non-negative"))
    v = zeros(T, 1 << n)
    v[1] = one(T)
    return StateVector{T}(v, n)
end
StateVector(::Type{T}, n::Integer) where {T<:Number} = StateVector{T}(n)
StateVector(n::Integer) = StateVector(ComplexF64, n)

"底层存储（长度 `2^n` 的一维振幅向量）。"
storage(x::StateVector) = x.data
_nqubits(x::StateVector) = x.n

Base.eltype(::Type{StateVector{T}}) where {T} = T
Base.eltype(x::StateVector) = eltype(typeof(x))
Base.zero(x::StateVector) = StateVector(zero(storage(x)), _nqubits(x))
Base.getindex(x::StateVector, j::Int) = storage(x)[j]
Base.setindex!(x::StateVector, v, j::Int) = setindex!(storage(x), v, j)
Base.length(x::StateVector) = length(storage(x))
Base.similar(x::StateVector) = StateVector(similar(storage(x)), _nqubits(x))
Base.convert(::Type{StateVector{T}}, x::StateVector) where {T<:Number} =
    StateVector(convert(Vector{T}, storage(x)), _nqubits(x))
Base.copy(x::StateVector) = StateVector(copy(storage(x)), _nqubits(x))

Base.isapprox(x::StateVector, y::StateVector; kwargs...) = isapprox(storage(x), storage(y); kwargs...)
Base.:(==)(x::StateVector, y::StateVector) = storage(x) == storage(y) && _nqubits(x) == _nqubits(y)

Base.:+(x::StateVector, y::StateVector) = StateVector(storage(x) + storage(y), _nqubits(x))
Base.:-(x::StateVector, y::StateVector) = StateVector(storage(x) - storage(y), _nqubits(x))
Base.:-(x::StateVector) = StateVector(-storage(x), _nqubits(x))
Base.:*(x::StateVector, y::Number) = StateVector(storage(x) * y, _nqubits(x))
Base.:*(x::Number, y::StateVector) = y * x
Base.:/(x::StateVector, y::Number) = StateVector(storage(x) / y, _nqubits(x))
Base.:*(m::AbstractMatrix, x::StateVector) = StateVector(m * storage(x), _nqubits(x))

LinearAlgebra.norm(x::StateVector) = norm(storage(x))
LinearAlgebra.dot(x::StateVector, y::StateVector) = dot(storage(x), storage(y))
LinearAlgebra.dot(x::StateVector, m::AbstractVector, y::StateVector) = dot(storage(x), m, storage(y))
LinearAlgebra.normalize!(x::StateVector) = (normalize!(storage(x)); x)
LinearAlgebra.normalize(x::StateVector) = StateVector(normalize(storage(x)), _nqubits(x))

"""
    zero_state([T=ComplexF64,] n) -> StateVector

`|0…0⟩`。
"""
zero_state(::Type{T}, n::Integer) where {T<:Number} = StateVector{T}(n)
zero_state(n::Integer) = zero_state(ComplexF64, n)

"""
    rand_state([T=ComplexF64,] n) -> StateVector

Haar 近似随机的纯态。
"""
function rand_state(::Type{T}, n::Integer) where {T<:Number}
    n >= 1 || throw(ArgumentError("number of qubits must be positive"))
    v = randn(T, 1 << n)
    v ./= norm(v)
    return StateVector(v, n)
end
rand_state(n::Integer) = rand_state(ComplexF64, n)

# ── 态编码 ────────────────────────────────────────────────────────────────────

"""
    onehot_encoding([T=ComplexF64,] bits::AbstractVector{Int}) -> StateVector

计算基态编码：`bits[i]` 为 qubit `i-1` 的取值（0/1），如 `onehot_encoding([1, 0])`
是 `|01⟩`（qubit 0 = 1，qubit 1 = 0）。
"""
function onehot_encoding(::Type{T}, bits::AbstractVector{Int}) where {T<:Number}
    isempty(bits) && throw(ArgumentError("input index is empty"))
    all(b -> b == 0 || b == 1, bits) || throw(ArgumentError("bits must be 0 or 1"))
    idx = 0
    for (i, b) in enumerate(bits)
        idx |= b << (i - 1)
    end
    v = zeros(T, 1 << length(bits))
    v[idx+1] = one(T)
    return StateVector(v, length(bits))
end
onehot_encoding(bits::AbstractVector{Int}) = onehot_encoding(ComplexF64, bits)

"""
    qubit_encoding([T=ComplexF64,] θs::AbstractVector{<:Real}) -> StateVector

直积态编码：qubit `i-1` 处于 `cos(πθ/2)|0⟩ + sin(πθ/2)|1⟩`。
"""
function qubit_encoding(::Type{T}, θs::AbstractVector{<:Real}) where {T<:Number}
    isempty(θs) && throw(ArgumentError("empty input"))
    v = one(T)
    for θ in reverse(θs)  # qubit n-1 在 kron 外侧（最高位），qubit 0 最低位
        v = kron(v, convert(Vector{T}, [cos(π * θ / 2), sin(π * θ / 2)]))
    end
    return StateVector(v, length(θs))
end
qubit_encoding(θs::AbstractVector{<:Real}) = qubit_encoding(ComplexF64, θs)

"""
    amplitude_encoding([T=ComplexF64,] v; nqubits=...) -> StateVector

振幅编码：向量 `v` 归一化后作为 `nqubits` 比特态的振幅。
"""
function amplitude_encoding(::Type{T}, v::AbstractVector{<:Number};
                            nqubits::Int=ceil(Int, log2(length(v)))) where {T<:Number}
    vn = norm(v)
    vn > 0 || throw(ArgumentError("input vector is zero"))
    isapprox(vn, 1.0) || @warn "input vector is not normalized; it will be renormalized as a quantum state."
    d = 1 << nqubits
    length(v) <= d || throw(DimensionMismatch("input length $(length(v)) exceeds 2^$nqubits"))
    vv = zeros(T, d)
    vv[1:length(v)] .= v ./ vn
    return StateVector(vv, nqubits)
end
amplitude_encoding(v::AbstractVector; kwargs...) = amplitude_encoding(ComplexF64, v; kwargs...)

# ── 就地重置 / 取分量 ─────────────────────────────────────────────────────────

"""
    reset!(x::StateVector) -> StateVector

重置为 `|0…0⟩`（就地）。
"""
function reset!(x::StateVector)
    fill!(storage(x), zero(eltype(x)))
    x[1] = one(eltype(x))
    return x
end

"""
    reset_onehot!(x::StateVector, bits::AbstractVector{Int}) -> StateVector

就地重置为计算基态（`bits[i]` = qubit `i-1` 的取值）。
"""
function reset_onehot!(x::StateVector, bits::AbstractVector{Int})
    _nqubits(x) == length(bits) || throw(ArgumentError("input basis mismatch with number of qubits"))
    idx = 0
    for (i, b) in enumerate(bits)
        idx |= b << (i - 1)
    end
    fill!(storage(x), zero(eltype(x)))
    x[idx+1] = one(eltype(x))
    return x
end

"""
    reset_qubit!(x::StateVector, θs::AbstractVector{<:Real}) -> StateVector

就地重置为 `qubit_encoding(θs)` 直积态。
"""
function reset_qubit!(x::StateVector, θs::AbstractVector{<:Real})
    _nqubits(x) == length(θs) || throw(ArgumentError("input basis mismatch with number of qubits"))
    copyto!(storage(x), storage(qubit_encoding(eltype(x), θs)))
    return x
end

"""
    amplitude(s::StateVector, bits::AbstractVector{Int}; scaling=1.0)

读取振幅 `⟨bits|s⟩`；`scaling = sqrt(2)` 时返回重标度版本（旧接口兼容）。
"""
function amplitude(s::StateVector, bits::AbstractVector{Int}; scaling::Real=1.0)
    _nqubits(s) == length(bits) || throw(ArgumentError("input basis mismatch with number of qubits"))
    idx = 0
    for (i, b) in enumerate(bits)
        idx |= b << (i - 1)
    end
    return scaling == 1 ? s[idx+1] : s[idx+1] * scaling^(_nqubits(s))
end

"全部振幅。"
amplitudes(s::StateVector) = storage(s)

"""
    permute(s::StateVector, perm::AbstractVector{Int}) -> StateVector

置换比特轴（1-based 比特号）：结果的 qubit `k` 为原来的 qubit `perm[k]`。
"""
function permute(s::StateVector, perm::AbstractVector{Int})
    n = _nqubits(s)
    length(perm) == n || throw(ArgumentError("perm length must be $n"))
    sort(collect(perm)) == collect(1:n) || throw(ArgumentError("perm must be a permutation of 1:$n"))
    n == 0 && return copy(s)
    v = permutedims(reshape(storage(s), ntuple(_ -> 2, n)), collect(perm))  # permutedims 用 1-based 轴号
    return StateVector(reshape(v, length(s)), n)
end

# ── 保真度 / 距离 ─────────────────────────────────────────────────────────────

"""
    fidelity(x, y)

纯态间为 `|⟨x|y⟩|²`；其他组合见 `DensityMatrix` 方法。
"""
fidelity(x::StateVector, y::StateVector) = abs2(dot(x, y))
distance2(x::StateVector, y::StateVector) = _distance2(x, y)
distance(x::StateVector, y::StateVector) = _distance(x, y)
