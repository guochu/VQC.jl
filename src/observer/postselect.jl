export QSelect, post_select, post_select!

divd(x::Number, y::Number) = x / y
@adjoint divd(x::Number, y::Number) = divd(x, y), z -> begin
    (z / conj(y), -z * conj(x / y^2) )
end


struct QSelect <: AbstractDifferentiableQuantumOperation
	key::Int
	state::Int

	function QSelect(key::Int, state::Int=0)
		(state==0 || state==1) || error("selected state must be 0 or 1.")
		new(key, state)
	end
end

scalar_type(s::QSelect) = Float64

function apply!(x::QSelect, qstate::AbstractVector)
	x.state == 0 ? _apply_gate_impl(x.key, UP, qstate) : _apply_gate_impl(x.key, DOWN, qstate)
	s = norm(qstate)
	qstate ./= s
	return s
end

function _apply_throw_impl(x::QSelect, qstate::AbstractVector)
	swap!(qstate, 1, x.key)
	ss = div(length(qstate), 2)
	tmp = reshape(qstate, (2, ss))
	r = x.state==0 ? ZERO'*tmp : ONE'*tmp
	swap!(qstate, 1, x.key)
	return reshape(r, length(r))
end

function _apply_keep_impl(x::QSelect, qstate::AbstractVector)
	tmp = copy(qstate)
	x.state == 0 ? _apply_gate_impl(x.key, UP, tmp) : _apply_gate_impl(x.key, DOWN, tmp)
	return tmp
end

function _apply_impl(x::QSelect, qstate::AbstractVector; keep::Bool=false)
	return keep ? _apply_keep_impl(x, qstate) : _apply_throw_impl(x, qstate)
end

cnorm(x::AbstractVector) = sqrt(vdot(conj(x), x))

function apply(x::QSelect, qstate::AbstractVector; keep::Bool=false)
	r = _apply_impl(x, qstate, keep=keep)
	ns = cnorm(r)
	return divd.(r, ns), real(ns)
end

"""
	post_select!(qstate::AbstractVector, key::Int, state::Int=0)
Post-select the i-th qubit of the quantum state, and the quantum state is updated inplace, the probability is returned.
"""
post_select!(qstate::AbstractVector, key::Int, state::Int=0) = apply!(QSelect(key, state), qstate)

"""
	post_select(qstate::AbstractVector, key::Int, state::Int=0; keep::Bool=false)
Return the collapsed quantum state as well as the probability.
"""
post_select(qstate::AbstractVector, key::Int, state::Int=0; keep::Bool=false) = apply(QSelect(key, state), qstate, keep=keep)

@adjoint QSelect(key::Int, state::Int) = QSelect(key, state), z -> (nothing, nothing)

@adjoint _apply_throw_impl(x::QSelect, qstate::AbstractVector) = _apply_throw_impl(x, qstate), z -> begin
    if x.state == 0
        m = kron(z, ZERO)
    else
    	m = kron(z, ONE)
    end
    swap!(m, 1, x.key)
    return (nothing, m)
end

@adjoint _apply_keep_impl(x::QSelect, qstate::AbstractVector) = _apply_keep_impl(x, qstate), z -> (nothing, _apply_keep_impl(x, z))

@adjoint _apply_impl(x::QSelect, qstate::AbstractVector; keep::Bool) = begin
	return keep ? Zygote.pullback(_apply_keep_impl, x, qstate) : Zygote.pullback(_apply_throw_impl, x, qstate)
    # if keep
    #     v, back = Zygote.pullback(_apply_keep_impl, x, qstate)
    # else
    # 	v, back = Zygote.pullback(_apply_throw_impl, x, qstate)
    # end
    # return v, z -> begin
    # 	a, b = back(z)
    # 	return (a, b, nothing)
    # end
end

