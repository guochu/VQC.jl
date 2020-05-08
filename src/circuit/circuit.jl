export QCircuit, add!, extend!, apply!

data(s::AbstractCircuit) = s.data
function scalar_type(s::AbstractCircuit)
	T = Float64
	for gate in s
	    T = promote_type(T, scalar_type(gate))
	end
	return T
end

# vector interface
Base.getindex(x::AbstractCircuit, i::Int) = Base.getindex(data(x), i)
Base.setindex!(x::AbstractCircuit, v::AbstractGate,  i::Int) = Base.setindex!(data(x), v, i)
Base.length(x::AbstractCircuit) = Base.length(data(x))
Base.iterate(x::AbstractCircuit) = Base.iterate(data(x))
Base.iterate(x::AbstractCircuit, state) = Base.iterate(data(x), state)
Base.eltype(x::AbstractCircuit) = Base.eltype(data(x))

# attributes
Base.isempty(x::AbstractCircuit) = Base.isempty(data(x))
Base.empty!(x::AbstractCircuit) = empty!(data(x))

"""
	add!(x::AbstractCircuit, s)
Push a new gate into the circuit.
"""
add!(x::AbstractCircuit, s) = push!(x, s)

"""
	Base.push!(x::AbstractCircuit, s::AbstractGate)
Push a new gate into the circuit.
"""
Base.push!(x::AbstractCircuit, s::AbstractGate) = Base.push!(data(x), s)
Base.push!(x::AbstractCircuit, s::Tuple{Int, M}) where M = push!(x, OneBodyGate(s[1], s[2]))
Base.push!(x::AbstractCircuit, s::Tuple{NTuple{1, Int}, M}) where M = push!(x, OneBodyGate(s[1][1], s[2]))
Base.push!(x::AbstractCircuit, s::Tuple{NTuple{2, Int}, M}) where M = push!(x, TwoBodyGate(s[1], s[2]))
Base.push!(x::AbstractCircuit, s::Tuple{NTuple{3, Int}, M}) where M = push!(x, ThreeBodyGate(s[1], s[2]))

Base.lastindex(x::AbstractCircuit) = Base.lastindex(data(x))

extend!(x::AbstractCircuit, s) = append!(x, s)

"""
	Base.append!(x::AbstractCircuit, y::AbstractCircuit)
Append a new circuit into an existing circuit.
"""
Base.append!(x::AbstractCircuit, y::AbstractCircuit) = append!(data(x), data(y))

"""
	Base.append!(x::AbstractCircuit, y::Vector{T}) where {T<:AbstractGate}
Append a new circuit into an existing circuit.
"""
Base.append!(x::AbstractCircuit, y::Vector{T}) where {T<:AbstractGate} = append!(data(x), y)

Base.transpose(x::AbstractCircuit) = typeof(x)([transpose(x[i]) for i = length(x):-1:1])
Base.conj(x::AbstractCircuit) = typeof(x)([conj(x[i]) for i = 1:length(x)])
Base.adjoint(x::AbstractCircuit) = typeof(x)([x[i]' for i = length(x):-1:1])
shift(x::AbstractCircuit, l::Int) = typeof(x)([shift(gate, l) for gate in x])



"""
Quantum circuit
"""

struct QCircuit <: AbstractCircuit
	data::Vector{AbstractGate}
end

QCircuit() = QCircuit(Vector{AbstractGate}())
Base.copy(x::QCircuit) = QCircuit(copy(data(x)))

pos_in_range(key::Int, n::Int) = key >= 1 && key <= n
function pos_in_range(key::NTuple{N, Int}, n::Int) where N
	for item in key
		pos_in_range(item, n) || return false
	end
	return true
end

function pos_in_range(circuit::AbstractCircuit, n::Int)
	for gate in circuit
		pos_in_range(key(gate), n) || return false
	end
	return true
end

"""
	apply!(circuit::AbstractCircuit, v::Vector)
Apply quantum circuit onto quantum state, the quantum state is updated inplace.
"""
function apply!(circuit::AbstractCircuit, v::Vector)
	pos_in_range(circuit, nqubits(v)) || error("key out of range.")
	for gate in circuit
		apply_gate!(gate, v)
	end
end

"""
	*(circuit::AbstractCircuit, v::AbstractVector)
Apply quantum circuit onto quantum state, the input quantum state is unchanged, return a new quantum state.
"""
function *(circuit::AbstractCircuit, v::AbstractVector)
	T = promote_type(scalar_type(circuit), scalar_type(v))
	v1 = Vector{T}(v)
	apply!(circuit, v1)
	return v1
end

"""
	*(v::AbstractVector, circuit::AbstractCircuit) = transpose(circuit) * v
Apply quantum circuit onto quantum state, in a reverse order, return a new quantum state.
"""
*(v::AbstractVector, circuit::AbstractCircuit) = transpose(circuit) * v
