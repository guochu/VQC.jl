using KrylovKit: eigsolve, exponentiate


function ground_state(h::QubitsOperator; kwargs...)
	ishermitian(h) || throw(ArgumentError("input operator is not hermitian."))
	n = QuantumCircuits.get_largest_pos(h)
	T = eltype(h)
	init_state = rand_state(T, n)

	eigvalues, eigvectors, info = eigsolve(x -> storage(h(StateVector(x, n))), storage(init_state), 1, :SR; ishermitian=true, kwargs...)
	(info.converged>=1) || error("eigsolve fails to converge.")
	return eigvalues[1], StateVector(eigvectors[1], n)
end


function time_evolution(h::QubitsOperator, t::Number, v::StateVector; kwargs...)
	ishermitian(h) || throw(ArgumentError("input operator is not hermitian."))
	(QuantumCircuits.get_largest_pos(h) <= nqubits(v)) || throw(ArgumentError("number of qubits mismatch."))
	n = nqubits(v)
	T = promote_type(eltype(h), typeof(t), eltype(v) )
	v = convert(StateVector{T}, v)
	tmp, info = exponentiate(x -> storage(h(StateVector(x, n))), t, storage(v); ishermitian=true, kwargs...)
	(info.converged>=1) || error("eigsolve fails to converge.")
	return StateVector(tmp, n)
end