export fock_state, boson, spin_half, spin_half_state


struct ElementaryParticle
    data::Dict{String, Array{T, 2} where T}
end

data(s::ElementaryParticle) = s.data

Base.get(s::ElementaryParticle, key::String) = get(data(s), key)
Base.getindex(s::ElementaryParticle, key::String) = data(s)[key]
get_identity(s::ElementaryParticle) = s["I"]
physical_dimension(s::ElementaryParticle) = size(get_identity(s), 1)

# _to_sparse(m) = Dict(k=>sparse(v) for (k, v) in m)

function fock_state(::Type{T}, d::Int, j::Int) where {T <: Number}
	(j >=0 && j <= d-1) || error("given state is out of range.")
	s = zeros(T, d)
	s[d - j] = 1
	return s
end

fock_state(d::Int, j::Int) = fock_state(Float64, d, j)

function boson(;d::Int=5)
    (d <= 1) && error("d must be larger than 1.")
    adag = zeros(Float64, d, d)
    for i = 1:(d - 1)
        adag[i, i+1] = sqrt(d-i)
    end
    a = Array(transpose(adag))
    n = adag * a
    n2 = n * n
    return ElementaryParticle(Dict("a"=>a, "adag"=>adag, "n"=>n, "n2"=>n2, "I"=>eye(Float64, d)))
end


spin_half() = ElementaryParticle(Dict("sx"=>X, "sy"=>Y, "sz"=>Z, "su"=>UP, "sd"=>DOWN, "sp"=>[0. 1.; 0. 0.],
"sm"=>[0. 0.; 1. 0.], "I"=>eye(Float64, 2)))

spin_half_state(j::Int) = begin
    (j == 0 || j == 1) || error("spin half state must be 0 or 1.")
    return j == 0 ? ZERO : ONE
end

