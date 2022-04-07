struct DensityMatrix{T <: Number}
	data::Vector{T}
	nqubits::Int

function DensityMatrix{T}(data::Vector{<:Number}, nqubits::Int) where {T <: Number}
	(length(data) == 2^(2*nqubits)) || throw(DimensionMismatch())
	new{T}(convert(Vector{T}, data), nqubits)
end

end

function DensityMatrix{T}(m::AbstractMatrix{<:Number}, nqubits::Int) where {T <: Number}
	(size(m, 1) == size(m, 2) == 2^nqubits) || throw(DimensionMismatch())
	return DensityMatrix{T}(reshape(m, length(m)), nqubits)
end
DensityMatrix(data::Union{AbstractVector, AbstractMatrix}, nqubits::Int) = DensityMatrix{eltype(data)}(data, nqubits)
DensityMatrix(data::AbstractVector) = DensityMatrix(data, div(_nqubits(data), 2))
DensityMatrix(data::AbstractMatrix) = DensityMatrix(reshape(data, length(data)))
function DensityMatrix{T}(nqubits::Int) where {T<:Number}
	 v = zeros(T, 2^(2*nqubits))
	 v[1,1] = 1
	 return DensityMatrix{T}(v, nqubits)
end
DensityMatrix(::Type{T}, nqubits::Int) where {T<:Number} = DensityMatrix{T}(nqubits)
DensityMatrix(nqubits::Int) = DensityMatrix(ComplexF64, nqubits)
DensityMatrix(x::DensityMatrix) = DensityMatrix(x.data, nqubits(x))
DensityMatrix(x::StateVector) = (x_data = storage(x); DensityMatrix(kron(conj(x_data), x_data), nqubits(x)))


storage(x::DensityMatrix) = (L = 2^(nqubits(x)); reshape(x.data, L, L))
QuantumCircuits.nqubits(x::DensityMatrix) = x.nqubits

Base.eltype(::Type{DensityMatrix{T}}) where T = T
Base.eltype(x::DensityMatrix) = eltype(typeof(x))
Base.getindex(x::DensityMatrix, j::Int...) = getindex(storage(x), j...)
Base.setindex!(x::StateVector, v, j::Int...) = setindex!(storage(x), v, j...)

Base.convert(::Type{DensityMatrix{T}}, x::DensityMatrix) where {T<:Number} = DensityMatrix(convert(Vector{T}, x.data), nqubits(x))
Base.copy(x::DensityMatrix) = DensityMatrix(copy(x.data), nqubits(x))


Base.cat(v::DensityMatrix) = v
function Base.cat(v::DensityMatrix...)
    a, b = _qcat_util(storage.(v)...)
    return DensityMatrix(kron(a, b))
end

Base.isapprox(x::DensityMatrix, y::DensityMatrix; kwargs...) = isapprox(x.data, y.data; kwargs...)
Base.:(==)(x::DensityMatrix, y::DensityMatrix) = x.data == y.data


LinearAlgebra.tr(x::DensityMatrix) = tr(storage(x))
LinearAlgebra.dot(x::DensityMatrix, y::DensityMatrix) = dot(storage(x), storage(y))
LinearAlgebra.normalize!(x::DensityMatrix) = (x.data ./= tr(x); x)
LinearAlgebra.normalize(x::DensityMatrix) = normalize!(copy(x))

distance2(x::DensityMatrix, y::DensityMatrix) = _distance2(x, y)
distance(x::DensityMatrix, y::DensityMatrix) = _distance(x, y)






