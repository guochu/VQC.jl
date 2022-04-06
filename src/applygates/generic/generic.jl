
include("serial_short_range.jl")
include("threaded_short_range.jl")
include("threaded_long_range.jl")


apply_serial!(x::Gate, s::AbstractVector) = _apply_gate_2!(ordered_positions(x), ordered_mat(x), s)


"""
    currently support mostly 5-qubit gate
"""
apply_threaded!(x::Gate, s::AbstractVector) = (length(s) >= 32) ? _apply_gate_threaded2!(ordered_positions(x), ordered_mat(x), s) : apply_serial!(x, s)

apply_serial!(x::Gate, state::StateVector) = (apply_serial!(x, storage(state)); state)
function apply!(x::Gate, state::StateVector) 
	@assert _check_pos_range(x, state)
	T = eltype(state)
	if (eltype(x) <: Complex) && (T <: Real)
		state = convert(StateVector{Complex{T}}, state)
	end
	apply_threaded!(x, storage(state))
	return state
end


function apply!(circuit::QCircuit, state::StateVector)
	for gate in circuit
		state = apply!(gate, state)
	end
	return state
end

Base.:*(circuit::QCircuit, state::StateVector) = apply!(circuit, copy(state))

function _check_pos_range(x::Gate, state::StateVector)
	pos = ordered_positions(x)
	(length(pos) > 5) && throw("only implement 5-qubit gates and less currently.")
	return (pos[1] >= 1) && (pos[end] <= nqubits(state))
end
