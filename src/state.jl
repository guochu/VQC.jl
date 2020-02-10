export qstate, qrandn, distance, vdot, onehot
export amplitude, amplitudes, probability, probabilities


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

function distance(x::AbstractVector, y::AbstractVector)
    sA = dot(x, x)
    sB = dot(y, y)
    c = dot(x, y)
    r = real(sA+sB-2*c)
    # println("$sA, $sB, $c")
    (r >= 0) || error("distance $r is negative.")
    return r
end 

# expectation(a::AbstractVector, circuit::AbstractCircuit, b::AbstractVector) = dot(a, circuit * b)

nqubits(s::AbstractVector) = begin
    n = round(Int, log2(length(s))) 
    (2^n == length(s)) || error("state can not be interpretted as a qubit state.")
    return n
end 

kernal_mapping(s::Real) = [cos(s*pi/2), sin(s*pi/2)]

function qstate(::Type{T}, mpsstr::AbstractVector{<:Real}) where {T <: Number}
	isempty(mpsstr) && error("no state")
	if length(mpsstr) == 1
	    v = kernal_mapping(mpsstr[1])
	else
		v = kron(kernal_mapping.(reverse(mpsstr))...)
	end
	return Vector{T}(v)
end

qstate(mpsstr::AbstractVector{<:Real}) = qstate(Complex{Float64}, mpsstr)
qstate(::Type{T}, n::Int) where {T <: Number} = qstate(T, [0 for _ in 1:n])
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

function amplitude(s::AbstractVector, i::AbstractVector{Int}) 
	(length(i)==nqubits(s)) || error("basis mismatch with number of qubits.")
	for s in i
	    (s == 0 || s == 1) || error("qubit state must be 0 or 1.")
	end
	cudim = dim2cudim_col(Tuple([2 for _ in 1:length(i)]))
	idx = mind2sind_col(i, cudim)
	return s[idx+1]
end

amplitudes(s::AbstractVector) = s

probabilities(s::AbstractVector) = (abs.(s)).^2

probability(s::AbstractVector, i::AbstractVector{Int}) = abs(amplitude(s, i))^2