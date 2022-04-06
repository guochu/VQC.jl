"""
	apply!(x::QMeasure, qstate::StateVector)
"""
function apply!(x::QMeasure, qstate::StateVector)
	x.keep || error("only keep mode is implemented.")
	probability, istate = _local_measure(storage(qstate), x.position)
	op = I₂
	if x.auto_reset
		m = kron(op[:, istate], QuantumCircuits.Gates.ZERO)/sqrt(probability)
	else
		m = kron(op[:, istate], op[:, istate])/sqrt(probability)
	end
	m = reshape(m, (2, 2))
	apply!(gate(x.position, m), qstate)
	return istate-1, probability
end


"""
	measure!(qstate::StateVector, pos::Int; auto_reset::Bool=true)
Measure the i-th qubit of the quantum state, and return a 2-tuple including the measurement outcome, \n 
and the probability with which we get the outcome. The quantum state is updated inplace. \n 
If auto_reset=true,the measured qubit is reset to 0, otherwise it will be the same as the measurement outcome.
"""
measure!(qstate::StateVector, pos::Int; auto_reset::Bool=true) = apply!(QMeasure(pos, auto_reset=auto_reset), qstate)


function apply(s::QMeasure, qstater::StateVector)
	qstate = storage(qstater)
	swap!(qstate, 1, s.position)
	probability, istate = _local_measure(qstate, 1)
	# println("with probability $probability to get $(istate-1)")
	ss = div(length(qstate), 2)
	r = (transpose(I₂[:, istate])*reshape(qstate, (2, ss)))/sqrt(probability) 
	swap!(qstate, 1, s.position)
	return StateVector(reshape(r, length(r)), nqubits(qstater)-1), istate-1, probability	
end

"""
	measure(qstate::StateVector, i::Int)
Measure the i-th qubit of the quantum state, and return a 3-tuple including the collapsed quantum state, \n 
the measurement outcome, and the probability with which we get the outcome.
"""
measure(qstate::StateVector, pos::Int) = apply(QMeasure(pos), qstate)

function compute_probability_zero(v::AbstractVector, key::Int)
	L = length(v)
	pos = 2^(key-1)
	stri = pos * 2
	r = 0.
	for i in 0:stri:(L-1)
		@inbounds for j in 0:(pos-1)
			l = i + j + 1
			r += abs2(v[l])
		end
	end
	return r
end


function _local_measure(v::AbstractVector, key::Int)
	p0 = compute_probability_zero(v, key)
	l = [p0, 1-p0]
	i = discrete_sample(l)
	return l[i], i
end
