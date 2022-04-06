

function apply!(x::QSelect, qstate::StateVector)
	x.state == 0 ? apply!(gate(x.position, QuantumCircuits.Gates.UP), qstate) : apply!(gate(x.position, QuantumCircuits.Gates.DOWN), qstate)
	s = norm(qstate)
	qstate.data ./= s
	return s
end

function apply(x::QSelect, qstate::StateVector; keep::Bool=false)
	r = _apply_impl(x, qstate, keep=keep)
	ns = cnorm(storage(r))
	return r / ns, real(ns)
end

"""
	post_select!(qstate::StateVector, key::Int, state::Int=0)
Post-select the i-th qubit of the quantum state, and the quantum state is updated inplace, the probability is returned.
"""
post_select!(qstate::StateVector, key::Int, state::Int=0) = apply!(QSelect(key, state), qstate)

"""
	post_select(qstate::StateVector, key::Int, state::Int=0; keep::Bool=false)
Return the collapsed quantum state as well as the probability.
"""
post_select(qstate::StateVector, key::Int, state::Int=0; keep::Bool=false) = apply(QSelect(key, state), qstate, keep=keep)





function _apply_throw_impl(x::QSelect, qstate::StateVector)
	swap!(storage(qstate), 1, x.position)
	ss = div(length(storage(qstate)), 2)
	tmp = reshape(storage(qstate), (2, ss))
	r = x.state==0 ? QuantumCircuits.Gates.ZERO'*tmp : QuantumCircuits.Gates.ONE'*tmp
	swap!(storage(qstate), 1, x.position)
	return StateVector(reshape(r, length(r)), nqubits(qstate)-1)
end

function _apply_keep_impl(x::QSelect, qstate::StateVector)
	tmp = copy(qstate)
	x.state == 0 ? apply!(gate(x.position, QuantumCircuits.Gates.UP), tmp) : apply!(gate(x.position, QuantumCircuits.Gates.DOWN), tmp)
	return StateVector(tmp, nqubits(qstate)-1)
end

function _apply_impl(x::QSelect, qstate::StateVector; keep::Bool=false)
	return keep ? _apply_keep_impl(x, qstate) : _apply_throw_impl(x, qstate)
end

cnorm(x::AbstractVector) = sqrt(dot(x, x))



@adjoint QSelect(key::Int, state::Int) = QSelect(key, state), z -> (nothing, nothing)

@adjoint _apply_throw_impl(x::QSelect, qstate::StateVector) = _apply_throw_impl(x, qstate), z -> begin
    if x.state == 0
        m = kron(z, ZERO)
    else
    	m = kron(z, ONE)
    end
    swap!(m, 1, x.position)
    return (nothing, m)
end

@adjoint _apply_keep_impl(x::QSelect, qstate::StateVector) = _apply_keep_impl(x, qstate), z -> (nothing, storage(_apply_keep_impl(x, StateVector(z, nqubits(qstate))) ))

@adjoint _apply_impl(x::QSelect, qstate::StateVector; keep::Bool) = begin
	return keep ? Zygote.pullback(_apply_keep_impl, x, qstate) : Zygote.pullback(_apply_throw_impl, x, qstate)
end
