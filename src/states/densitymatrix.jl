# densitymatrix.jl — 混合态（密度矩阵）
#
# 核心层：不依赖 QuantumCircuits 的任何类型。

"""
密度矩阵：长度 `2^(2n)` 的一维存储（列 major 展开的 `2^n × 2^n` 矩阵，
行索引 = ket，小端序）。

    DensityMatrix(mat, n)         # d×d 矩阵 + 比特数
    DensityMatrix(vec, n)         # 平坦存储 + 比特数
    DensityMatrix(mat)            # 自动推断
    DensityMatrix(ComplexF64, n)  # |0…0⟩⟨0…0|
    DensityMatrix(ψ::StateVector) # 纯态的密度矩阵
"""
struct DensityMatrix{T<:Number}
    data::Vector{T}
    n::Int

    function DensityMatrix{T}(data::AbstractVector{<:Number}, n::Integer) where {T<:Number}
        n >= 0 || throw(ArgumentError("number of qubits must be non-negative"))
        length(data) == 1 << (2 * n) ||
            throw(DimensionMismatch("expected length $(1 << (2n)) for $n qubits, got $(length(data))"))
        new{T}(convert(Vector{T}, data), Int(n))
    end
end

function DensityMatrix{T}(m::AbstractMatrix{<:Number}, n::Integer) where {T<:Number}
    size(m, 1) == size(m, 2) == 1 << n || throw(DimensionMismatch("expected $(1 << n)×$(1 << n) matrix"))
    return DensityMatrix{T}(reshape(m, :), n)
end
DensityMatrix(data::AbstractVector{<:Number}, n::Integer) = DensityMatrix{eltype(data)}(data, n)
DensityMatrix(m::AbstractMatrix{<:Number}, n::Integer) = DensityMatrix{eltype(m)}(m, n)
function DensityMatrix(data::AbstractVector{<:Number})
    L = length(data)
    (L >= 4 && L & (L - 1) == 0) || throw(ArgumentError("length must be a power of four, got $L"))
    return DensityMatrix(data, Int(log2(L)) ÷ 2)
end
DensityMatrix(m::AbstractMatrix{<:Number}) = DensityMatrix(reshape(m, :))
function DensityMatrix{T}(n::Integer) where {T<:Number}
    v = zeros(T, 1 << (2 * n))
    v[1] = one(T)
    return DensityMatrix{T}(v, n)
end
DensityMatrix(::Type{T}, n::Integer) where {T<:Number} = DensityMatrix{T}(n)
DensityMatrix(n::Integer) = DensityMatrix(ComplexF64, n)
DensityMatrix(x::StateVector) =
    DensityMatrix(kron(conj(storage(x)), storage(x)), _nqubits(x))

"""
`2^n × 2^n` 矩阵视图（与底层存储共享内存）。
"""
storage(x::DensityMatrix) = reshape(x.data, 1 << x.n, 1 << x.n)
_nqubits(x::DensityMatrix) = x.n

Base.eltype(::Type{DensityMatrix{T}}) where {T} = T
Base.eltype(x::DensityMatrix) = eltype(typeof(x))
Base.zero(x::DensityMatrix) = DensityMatrix(zero(x.data), _nqubits(x))
Base.getindex(x::DensityMatrix, j::Int...) = storage(x)[j...]
Base.setindex!(x::DensityMatrix, v, j::Int...) = setindex!(storage(x), v, j...)
Base.similar(x::DensityMatrix) = DensityMatrix(similar(x.data), _nqubits(x))
Base.convert(::Type{DensityMatrix{T}}, x::DensityMatrix) where {T<:Number} =
    DensityMatrix(convert(Vector{T}, x.data), _nqubits(x))
Base.copy(x::DensityMatrix) = DensityMatrix(copy(x.data), _nqubits(x))

Base.isapprox(x::DensityMatrix, y::DensityMatrix; kwargs...) = isapprox(x.data, y.data; kwargs...)
Base.:(==)(x::DensityMatrix, y::DensityMatrix) = x.data == y.data && _nqubits(x) == _nqubits(y)

Base.:+(x::DensityMatrix, y::DensityMatrix) = DensityMatrix(x.data + y.data, _nqubits(x))
Base.:-(x::DensityMatrix, y::DensityMatrix) = DensityMatrix(x.data - y.data, _nqubits(x))
Base.:-(x::DensityMatrix) = DensityMatrix(-x.data, _nqubits(x))
Base.:*(x::DensityMatrix, y::Number) = DensityMatrix(x.data * y, _nqubits(x))
Base.:*(x::Number, y::DensityMatrix) = y * x
Base.:/(x::DensityMatrix, y::Number) = DensityMatrix(x.data / y, _nqubits(x))

LinearAlgebra.tr(x::DensityMatrix) = tr(storage(x))
LinearAlgebra.dot(x::DensityMatrix, y::DensityMatrix) = dot(storage(x), storage(y))
LinearAlgebra.dot(x::DensityMatrix, m::AbstractMatrix, y::DensityMatrix) = dot(storage(x), m, storage(y))
LinearAlgebra.normalize!(x::DensityMatrix) = (x.data ./= tr(x); x)
LinearAlgebra.normalize(x::DensityMatrix) = normalize!(copy(x))
LinearAlgebra.ishermitian(x::DensityMatrix) = ishermitian(storage(x))
LinearAlgebra.isposdef(x::DensityMatrix) = isposdef(storage(x))

"""
    fidelity(x::DensityMatrix, y::DensityMatrix)

`tr(√x √y)`（Uhlmann 保真度的平方根约定见文档）。
"""
function fidelity(x::DensityMatrix, y::DensityMatrix)
    _nqubits(x) == _nqubits(y) || throw(ArgumentError("qubit number mismatch"))
    return real(tr(sqrt(Hermitian(storage(x))) * sqrt(Hermitian(storage(y)))))
end
fidelity(x::DensityMatrix, y::StateVector) = real(dot(storage(y), storage(x), storage(y)))
fidelity(x::StateVector, y::DensityMatrix) = fidelity(y, x)
distance2(x::DensityMatrix, y::DensityMatrix) = _distance2(x, y)
distance(x::DensityMatrix, y::DensityMatrix) = _distance(x, y)

"""
    schmidt_numbers(x::DensityMatrix)

密度矩阵的特征值（按降序），可作为谱分解 / 纠缠度量的输入。
"""
schmidt_numbers(x::DensityMatrix) = eigvals(Hermitian(storage(x)))
renyi_entropy(x::DensityMatrix; kwargs...) = renyi_entropy(real.(schmidt_numbers(x)); kwargs...)

"""
    rand_densitymatrix([T=ComplexF64,] n) -> DensityMatrix

随机（Wishart 构造归一化）密度矩阵。
"""
function rand_densitymatrix(::Type{T}, n::Integer) where {T<:Number}
    n >= 1 || throw(ArgumentError("number of qubits must be positive"))
    v = randn(T, 1 << n, 1 << n)
    rho = v' * v
    return DensityMatrix(rho / tr(rho), n)
end
rand_densitymatrix(n::Integer) = rand_densitymatrix(ComplexF64, n)

"""
    permute(s::DensityMatrix, perm::AbstractVector{Int}) -> DensityMatrix

置换比特轴（1-based 比特号）：结果的 qubit `k` 为原来的 qubit `perm[k]`。
"""
function permute(s::DensityMatrix, perm::AbstractVector{Int})
    n = _nqubits(s)
    length(perm) == n || throw(ArgumentError("perm length must be $n"))
    sort(collect(perm)) == collect(1:n) || throw(ArgumentError("perm must be a permutation of 1:$n"))
    n == 0 && return copy(s)
    p1 = collect(perm)  # permutedims 用 1-based 轴号
    vt = permutedims(reshape(s.data, ntuple(_ -> 2, 2n)), vcat(p1, p1 .+ n))
    return DensityMatrix(reshape(vt, length(s.data)), n)
end
