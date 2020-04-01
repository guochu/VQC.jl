export add!, Hamiltonian, matrix

mutable struct Hamiltonian
	particles::Vector{ElementaryParticle}
	h::AbstractMatrix
end

particles(s::Hamiltonian) = s.particles
get_identities(s::Hamiltonian) = get_identity.(particles(s))

matrix(s::Hamiltonian) = s.h
nsites(s::Hamiltonian) = length(particles(s))

function Hamiltonian(pts::Vector{ElementaryParticle})
	isempty(pts) && error("no particles.")
	d = prod(physical_dimension.(pts))
	return Hamiltonian(pts, spzeros(Float64, d, d))
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












