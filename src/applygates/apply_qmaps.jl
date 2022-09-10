


"""
	apply!(x::QuantumMap, state::DensityMatrix) 
	apply a generc quantum channel on the quantum state
"""
function apply!(x::QuantumMap, state::DensityMatrix) 
	@assert _check_pos_range(x, nqubits(state))
	T = eltype(state)
	if (eltype(x) <: Complex) && (T <: Real)
		state = convert(DensityMatrix{Complex{T}}, state)
	end
	_qmap_apply_threaded!(x, state.data, nqubits(state))
	return state
end

apply!(x::QuantumMap, state::StateVector) = apply!(x, DensityMatrix(state)) 

"""
	apply_dagger!(x::QuantumMap, state::DensityMatrix) 
	apply the hermitian conjugate of quantum map on rho
	this function is needed for auto-differentiation
"""
function apply_dagger!(x::QuantumMap, state::DensityMatrix) 
	@assert _check_pos_range(x, nqubits(state))
	T = eltype(state)
	if (eltype(x) <: Complex) && (T <: Real)
		state = convert(DensityMatrix{Complex{T}}, state)
	end
	_qmap_apply_dagger_threaded!(x, state.data, nqubits(state))
	return state
end

"""
	apply_inverse!(x::QuantumMap, state::DensityMatrix) 
	apply the inverse of quantum map on rho, this requires the supermatrix to be invertable.
	this function is needed for auto-differentiation
"""
function apply_inverse!(x::QuantumMap, state::DensityMatrix) 
	@assert _check_pos_range(x, nqubits(state))
	T = eltype(state)
	if (eltype(x) <: Complex) && (T <: Real)
		state = convert(DensityMatrix{Complex{T}}, state)
	end
	_qmap_apply_inverse_threaded!(x, state.data, nqubits(state))
	return state
end


# unitary gate operation on density matrix
function _qmap_apply_threaded!(x::QuantumMap{N}, s::AbstractVector, n::Int) where N
	pos = ordered_positions(x)
	pos2 = ntuple(i->pos[i]+n, N)
	all_pos = (pos..., pos2...)
	m = ordered_supermat(x)
	if length(s) >= 32
		_apply_gate_threaded2!(all_pos, m, s)
	else
		_apply_gate_2!(all_pos, m, s)
	end
end

function _qmap_apply_dagger_threaded!(x::QuantumMap{N}, s::AbstractVector, n::Int) where N
	pos = ordered_positions(x)
	pos2 = ntuple(i->pos[i]+n, N)
	all_pos = (pos..., pos2...)
	m = ordered_supermat(x)
	if length(s) >= 32
		_apply_gate_threaded2!(all_pos, m', s)
	else
		_apply_gate_2!(all_pos, m', s)
	end
end

function _qmap_apply_inverse_threaded!(x::QuantumMap{N}, s::AbstractVector, n::Int) where N
	pos = ordered_positions(x)
	pos2 = ntuple(i->pos[i]+n, N)
	all_pos = (pos..., pos2...)
	m = ordered_supermat(x)
	if length(s) >= 32
		_apply_gate_threaded2!(all_pos, inv(m), s)
	else
		_apply_gate_2!(all_pos, inv(m), s)
	end
end
