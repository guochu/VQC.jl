export Observers, todict

struct ObserverBase{M}
	ob::M
	name::String
	isherm::Bool
end


ObserverBase(m::AbstractMatrix, name::String) = ObserverBase(m, name, ishermitian(m))

is_hermitian(x::ObserverBase)  = x.isherm
name(x::ObserverBase) = x.name

"""
	ob can be an mpo, or a Dict{Int, Matrix}, or a function
	if ob is a function of the quantum state
"""
get_ob(x::ObserverBase) = x.ob


function push_result(x::ObserverBase, result::Vector, v::Number)
	if is_hermitian(x)
		iv = imag(v)
		(abs(iv) > OBSERVER_IMAG_TOL) && @warn "Imaginary part of hermitian observer $(name(x)) is $iv.\n"
		v = real(v)
	end
	push!(result, (name(x), v)) 
end

function apply_and_collect(x::ObserverBase, state::AbstractVector, result::Vector)
	v = state' * get_ob(x) * state
	push_result(x, result, v)
end


struct Observers 
	particles::Vector{ElementaryParticle}
	data::Vector{<:ObserverBase}
end

data(s::Observers) = s.data
particles(s::Observers) = s.particles
Base.isempty(s::Observers) = isempty(data(s))
Base.length(s::Observers) = length(data(s))


compute_norm(state::AbstractVector) = norm(state)
function apply_and_collect(x::Observers, state, result::Vector)
	nrm = compute_norm(state)
	push!(result, ("norm", nrm))
	for observer in data(x)
		apply_and_collect(observer, state, result)
	end
end

add!(s::Observers, key, op; name::String) = add!(s, Dict(k=>v for (k, v) in zip(key, op)), name=name)

add!(x::Observers, opstr::AbstractDict{Int}; name::String) = begin
	ps = particles(x)
	idens = get_identity.(ps)
    opstr = Dict(k=>ps[k][v] for (k, v) in opstr)
    m = generateprodham(idens, opstr)
    push!(data(x), ObserverBase(m, name))
end 

Observers(ps::Vector{ElementaryParticle}) = Observers(ps, Vector{ObserverBase}())
Observers(h::Hamiltonian) = Observers(particles(h))


function apply(s::Observers, state::AbstractVector)
	result = []
	apply_and_collect(s, state, result)
	return result
end


function todict(x::Vector)
	r = Dict{String, Vector{Number}}()
	for (key, value) in x
		v = Base.get!(r, key, Vector{Number}())
		Base.push!(v, value)
	end
	return Dict(k=>[v...] for (k, v) in r)
end

