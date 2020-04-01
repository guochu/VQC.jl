export qstate, qrandn, distance, distance2, vdot, cdot, onehot
export amplitude, amplitudes, probability, probabilities

cdot(x::AbstractArray, y::AbstractArray) = dot(x, y)

vdot(x::AbstractVector, y::AbstractVector) = begin
    (length(x)==length(y)) || error("quantum state size mismatch.")
    r = 0.
    for i in 1:length(x)
        r += x[i] * y[i]
    end
    return r
end

# cnorm(x::AbstractVector) = sqrt(dot(x, x))

renormalize!(s::AbstractVector) = begin
    sn = norm(s)
    tol = 1.0e-12
    sn <= tol && error("quantum state norm is small than $tol")
    s.data ./= sn
end 

renormalize(s::AbstractVector) = s / norm(s)

function distance2(x::AbstractVector, y::AbstractVector)
    sA = dot(x, x)
    sB = dot(y, y)
    c = dot(x, y)
    r = real(sA+sB-2*c)
    return abs(r)
end 

distance(x::AbstractVector, y::AbstractVector) = sqrt(distance2(x, y))

# expectation(a::AbstractVector, circuit::AbstractCircuit, b::AbstractVector) = dot(a, circuit * b)

nqubits(s::AbstractVector) = begin
    n = round(Int, log2(length(s))) 
    (2^n == length(s)) || error("state can not be interpretted as a qubit state.")
    return n
end 

kernal_mapping(s::Real) = [cos(s*pi/2), sin(s*pi/2)]

"""
	qstate(::Type{T}, thetas::AbstractVector{<:Real}) where {T <: Number}
Return a product quantum state of [[cos(pi*theta/2), sin(pi*theta/2)]] for theta in thetas]\n
Example: qstate(Complex{Float64}, [0.5, 0.7])
"""
function qstate(::Type{T}, mpsstr::AbstractVector{<:Real}) where {T <: Number}
	isempty(mpsstr) && error("no state")
	if length(mpsstr) == 1
	    v = kernal_mapping(mpsstr[1])
	else
		v = kron(kernal_mapping.(reverse(mpsstr))...)
	end
	return Vector{T}(v)
end

"""
	qstate(thetas::AbstractVector{<:Real}) = qstate(Complex{Float64}, thetas)
Return a product quantum state of [[cos(pi*theta/2), sin(pi*theta/2)]] for theta in thetas]
"""
qstate(mpsstr::AbstractVector{<:Real}) = qstate(Complex{Float64}, mpsstr)

"""
	qstate(::Type{T}, n::Int) where {T <: Number}
Return a product quantum state of [[1, 0] for _ in 1:n] for theta in thetas]
"""
qstate(::Type{T}, n::Int) where {T <: Number} = qstate(T, [0 for _ in 1:n])

"""
	qstate(n::Int) = qstate(Complex{Float64}, n)
Return a product quantum state of [[1, 0] for _ in 1:n] for theta in thetas]
"""
qstate(n::Int) = qstate(Complex{Float64}, n)

function onehot(::Type{T}, L::Int, pos::Int) where T
	r = zeros(T, L)
	r[pos] = 1
	return r
end

onehot(L::Int, pos::Int) = onehot(Complex{Float64}, L, pos)

function qrandn(::Type{T}, n::Int) where {T <: Number} 
	(n >= 1) || error("number of qubits must be positive.")
	v = randn(T, 2^n)
	v ./= norm(v)
	return v
end

qrandn(n::Int) = qrandn(Complex{Float64}, n)

"""
	amplitude(s::AbstractVector, i::AbstractVector{Int}) 
Return a single amplitude of the quantum state
"""
function amplitude(s::AbstractVector, i::AbstractVector{Int}) 
	(length(i)==nqubits(s)) || error("basis mismatch with number of qubits.")
	for s in i
	    (s == 0 || s == 1) || error("qubit state must be 0 or 1.")
	end
	cudim = dim2cudim_col(Tuple([2 for _ in 1:length(i)]))
	idx = mind2sind_col(i, cudim)
	return s[idx+1]
end

"""
	amplitudes(s::AbstractVector)
Return all amplitudes of the quantum state
"""
amplitudes(s::AbstractVector) = s

probabilities(s::AbstractVector) = (abs.(s)).^2

probability(s::AbstractVector, i::AbstractVector{Int}) = abs(amplitude(s, i))^2