export add!, Hamiltonian, matrix

# function _parse_op(op::AbstractString)
# 	if op == "X"
# 	    return X
# 	elseif op == "Y"
# 		return Y
# 	elseif op == "Z"
# 		return Z
# 	elseif op == "0"
# 		return UP
# 	elseif op == "1"
# 		return DOWN
# 	else
# 		error("unknown operation $op.")
# 	end
# end

# _parse_op(op::AbstractMatrix) = begin
#     (size(op) == (2,2)) || error("matrix shape error.")
#     return op
# end 

# _parse_op(op::AbstractDict) = Dict(k=>_parse_op(v) for (k, v) in op)

# function _generateprodhamimpl(L::Int, opstr)
# 	I2 = eye(2)
# 	vl = Vector{Any}(undef, L)
# 	for i in 1:L
# 	    v = get(opstr, i, nothing)
# 	    if v === nothing
# 	        vl[i] = I2
# 	    else
# 	    	vl[i] = v
# 	    end
# 	end
# 	return sparse(kron(vl...))
# end

# generateprodham(L::Int) = begin
#     n = 2^L
#     return spzeros(n, n)
# end

# generateprodham(L::Int, opstr::AbstractDict) = _generateprodhamimpl(L, _parse_op(opstr))

# _generateprodhamimpl(L::Int, key::AbstractVector, op::AbstractVector) = generateprodham(L, 
# 	Dict(k=>v for (k, v) in zip(key, op)))

# generateprodham(L::Int, key, op) = _generateprodhamimpl(L, [key...], [op...])

mutable struct Hamiltonian
	particles::Vector{ElementaryParticle}
	h::SparseMatrixCSC{Complex{Float64}, Int}
end

particles(s::Hamiltonian) = s.particles
get_identities(s::Hamiltonian) = get_identity.(particles(s))

matrix(s::Hamiltonian) = s.h
nsites(s::Hamiltonian) = length(particles(s))

function Hamiltonian(pts::Vector{ElementaryParticle})
	isempty(pts) && error("no particles.")
	d = prod(physical_dimension.(pts))
	return Hamiltonian(pts, spzeros(Complex{Float64}, d, d))
end

Hamiltonian(L::Int) = Hamiltonian([spin_half() for i in 1:L])



_normal_op(s::Hamiltonian, key::Int, op::String) = particles(s)[key][op]
_normal_op(s::Hamiltonian, key::Int, op::AbstractMatrix) = begin
    size(get_identity(particles(s)[key]) == size(op)) || error("wrong matrix size.")
    return op
end

_normal_op(s::Hamiltonian, opstr::AbstractDict) = Dict(k=>_normal_op(s, k, v) for (k, v) in opstr)

add!(s::Hamiltonian, opstr::AbstractDict; coeff::Number=1.) = (s.h += coeff*generateprodham(get_identities(s), _normal_op(s, opstr)))
add!(s::Hamiltonian, key, op; coeff::Number=1.) = add!(s, Dict(k=>v for (k, v) in zip(key, op)), coeff=coeff)

*(m::Hamiltonian, s::Number) = Hamiltonian(particles(m), matrix(m) * s)
*(s::Number, m::Hamiltonian) = m * s

+(x::Hamiltonian, y::Hamiltonian) = Hamiltonian(particles(x), matrix(x) + matrix(y))
-(x::Hamiltonian, y::Hamiltonian) = Hamiltonian(particles(x), matrix(x) - matrix(y))

*(m::Hamiltonian, v::AbstractVector) = matrix(m) * v
*(v::AbstractVector, m::Hamiltonian) = v * matrix(m)

Base.show(io::IO, x::Hamiltonian) = Base.show(io, matrix(x))












