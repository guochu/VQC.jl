struct StateVector{T <: Number} 
    data::Vector{T}
    nqubits::Int

function StateVector{T}(data::AbstractVector{<:Number}, nqubits::Int) where {T <: Number}
    @assert length(data) == 2^nqubits
    new{T}(convert(Vector{T}, data), nqubits)
end

end

StateVector(data::AbstractVector{T}, nqubits::Int) where {T <: Number} = StateVector{T}(data, nqubits)
StateVector{T}(data::AbstractVector{<:Number}) where T = StateVector{T}(data, _nqubits(data))
StateVector(data::AbstractVector{T}) where {T <: Number} = StateVector{T}(data)
function StateVector{T}(nqubits::Int) where {T<:Number} 
    v = zeros(T, 2^nqubits)
    v[1] = 1
    return StateVector{T}(v, nqubits)
end
StateVector(::Type{T}, nqubits::Int) where {T<:Number} = StateVector{T}(nqubits)
StateVector(nqubits::Int) = StateVector(ComplexF64, nqubits)
StateVector(x::StateVector) = StateVector(x.data, nqubits(x))

storage(x::StateVector) = x.data
QuantumCircuits.nqubits(x::StateVector) = x.nqubits

Base.eltype(::Type{StateVector{T}}) where T = T
Base.eltype(x::StateVector) = eltype(typeof(x))
Base.getindex(x::StateVector, j::Int) = getindex(storage(x), j)
Base.setindex!(x::StateVector, v, j::Int) = setindex!(storage(x), v, j)

Base.convert(::Type{StateVector{T}}, x::StateVector) where {T<:Number} = StateVector(convert(Vector{T}, storage(x)), nqubits(x))
Base.copy(x::StateVector) = StateVector(copy(storage(x)), nqubits(x))

Base.cat(v::StateVector) = v
function Base.cat(v::StateVector...)
    a, b = _qcat_util(storage.(v)...)
    return StateVector(kron(a, b))
end

Base.isapprox(x::StateVector, y::StateVector; kwargs...) = isapprox(storage(x), storage(y); kwargs...)
Base.:(==)(x::StateVector, y::StateVector) = storage(x) == storage(y)


Base.:+(x::StateVector, y::StateVector) = StateVector(storage(x) + storage(y), nqubits(x))
Base.:-(x::StateVector, y::StateVector) = StateVector(storage(x) - storage(y), nqubits(x))
Base.:*(x::StateVector, y::Number) = StateVector(storage(x) * y, nqubits(x))
Base.:*(x::Number, y::StateVector) = y * x
Base.:/(x::StateVector, y::Number) = StateVector(storage(x) / y, nqubits(x))
Base.:*(m::AbstractMatrix, x::StateVector) = StateVector( m * storage(x), nqubits(x) )


LinearAlgebra.norm(x::StateVector) = norm(storage(x))
LinearAlgebra.dot(x::StateVector, y::StateVector) = dot(storage(x), storage(y))
LinearAlgebra.normalize!(x::StateVector) = (normalize!(storage(x)); x)
LinearAlgebra.normalize(x::StateVector) = StateVector(normalize(storage(x)), nqubits(x))

"""
    fidelity(x, y) 
    tr(√x * √y) if x and y are density matrices
    ⟨x|y⟩^2 if x and y are pure states
"""
fidelity(x::StateVector, y::StateVector) = abs2(dot(x, y))
distance2(x::StateVector, y::StateVector) = _distance2(x, y)
distance(x::StateVector, y::StateVector) = _distance(x, y)


# encoding
onehot_encoding(::Type{T}, n::Int) where {T <: Number} = StateVector(onehot(T, 2^n, 1), n)
onehot_encoding(n::Int) = onehot_encoding(ComplexF64, n)
onehot_encoding(::Type{T}, i::AbstractVector{Int}) where {T <: Number} = StateVector(onehot(T, 2^(length(i)), _sub2ind(i)+1), length(i))
onehot_encoding(i::AbstractVector{Int})= onehot_encoding(ComplexF64, i)

function onehot(::Type{T}, L::Int, pos::Int) where T
    r = zeros(T, L)
    r[pos] = one(T)
    return r
end


"""
    kernal_mapping(s::Real) = [cos(s*pi/2), sin(s*pi/2)]
    This maps 0 -> [1, 0] (|0>), and 1 -> [0, 1] (|1>)
"""
kernal_mapping(s::Real) = [cos(s*pi/2), sin(s*pi/2)]


"""
    qstate(::Type{T}, thetas::AbstractVector{<:Real}) where {T <: Number}
Return a product quantum state of [[cos(pi*theta/2), sin(pi*theta/2)]] for theta in thetas]\n
Example: qstate(Complex{Float64}, [0.5, 0.7])
"""
function qubit_encoding(::Type{T}, i::AbstractVector{<:Real}) where {T <: Number}
    isempty(i) && throw("empty input.")
    v = [convert(Vector{T}, item) for item in kernal_mapping.(i)]
    (length(v) == 1) && return StateVector{T}(v[1])
    a, b = _qcat_util(v...)
    return StateVector(kron(a, b))
end  
qubit_encoding(mpsstr::AbstractVector{<:Real}) = qubit_encoding(ComplexF64, mpsstr)


function reset!(x::StateVector)
    fill!(storage(x), zero(eltype(x)))
    x[1] = one(eltype(x))
    return x
end
function reset_onehot!(x::StateVector, i::AbstractVector{Int})
    @assert nqubits(x) == length(i)
    pos = _sub2ind(i) + 1
    fill!(storage(x), zero(eltype(x)))
    x[pos] = one(eltype(x))
    return x
end
# reset!(x::StateVector, i::AbstractVector{Int}) = reset_onehot!(x, i)
function reset_qubit!(x::StateVector, i::AbstractVector{<:Real})
    @assert nqubits(x) == length(i)
    if length(i) == 1
        copyto!(storage(x), kernal_mapping(i[1]))
        return x
    end
    a, b = _qcat_util(kernal_mapping.(i)...)
    m = length(a)
    n = length(b)
    xs = storage(x)
    for j in 1:m
        n_start = (j-1) * n + 1
        n_end = j * n
        tmp = a[j]
        @. xs[n_start:n_end] = tmp * b
    end
    return x
end

function amplitude(s::StateVector, i::AbstractVector{Int}; scaling::Real=sqrt(2))
    @assert length(i)==nqubits(s)
    idx = _sub2ind(i)
    return scaling==1 ? s[idx] : s[idx] * scaling^(nqubits(s))
end
amplitudes(s::StateVector) = storage(s)

function rand_state(::Type{T}, n::Int) where {T <: Number}
    (n >= 1) || error("number of qubits must be positive.")
    v = randn(T, 2^n)
    v ./= norm(v)
    return StateVector(v, n)
end
rand_state(n::Int) = rand_state(ComplexF64, n)


function _sub2ind(v::AbstractVector{Int})
    @assert _is_valid_indices(v)
    isempty(v) && error("input index is empty.")
    L = length(v)
    r = v[1]
    for i in 2:L
        r |= v[i] << (i-1)
    end
    return r
end

function _qcat_util(vr::Union{AbstractVector, AbstractMatrix}...)
    v = reverse(vr)
    L = length(v)
    # println("$(typeof(v)), $L")
    (L >= 2) || error("something wrong.")
    Lh = div(L, 2)
    a = v[1]
    for i in 2:Lh
        a = kron(a, v[i])
    end
    b = v[Lh + 1]
    for i in Lh+2 : L
        b = kron(b, v[i])
    end
    return a, b
end

function _is_valid_indices(i::AbstractVector{Int})
    for s in i
        (s == 0 || s == 1) || return false 
    end   
    return true
end

_nqubits(s::AbstractVector) = begin
    n = round(Int, log2(length(s)))
    (2^n == length(s)) || error("state can not be interpretted as a qubit state.")
    return n
end



# function reset!(x::AbstractVector, i::AbstractVector{<:AbstractFloat})
#     (nqubits(x) == length(i)) || error("input basis mismatch with number of qubits.")
#     mpsstr = kernal_mapping.(i)
#     for (pos, item) in enumerate(Iterators.product(mpsstr...))
#         x[pos] = prod(item)
#     end
#     return x
# end


# function qrandn(::Type{T}, n::Int) where {T <: Number}
# 	(n >= 1) || error("number of qubits must be positive.")
# 	v = randn(T, 2^n)
# 	v ./= norm(v)
# 	return v
# end

# qrandn(n::Int) = qrandn(Complex{Float64}, n)
