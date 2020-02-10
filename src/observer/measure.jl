export QMeasure, apply, apply!, measure, measure!

function discrete_sample(l::Vector{Float64})
	isempty(l) && error("no results.")
	s = sum(l)
	L = length(l)
	l1 = Vector{Float64}(undef, L+1)
	l1[1] = 0
	for i=1:L
	    l1[i+1] = l1[i] + l[i]/s
	end
	s = rand(Float64)
	for i = 1:L
	    if (s >= l1[i] && s < l1[i+1])
	        return i
	    end
	end
	error("something wrong.")
end

function _local_measure(rhov::AbstractVector, basis::AbstractMatrix)
	d1 = size(basis, 1)
	d2 = div(length(rhov), d1)
	rho = reshape(rhov, (d1, d2))
	m = rho*rho'
	l = Float64[]
	d = size(basis, 2)
	tol = 1.0e-6
	for i = 1:d
		s = Base.transpose(basis[:, i])*m*conj(basis[:, i]) 
		sm = imag(s)
		(abs(sm) > tol) && @warn "Imaginary part of the measure result is larger than $tol.\n"
		push!(l, real(s))
	end
	i = discrete_sample(l)
	return l[i], i
end

struct QMeasure <: AbstractQuantumOperation
	key::Int
	auto_reset::Bool
	op::Array{Float64, 2}

	function QMeasure(key::Int; auto_reset::Bool=true)
		op = [1. 0.; 0. 1.]
		new(key, auto_reset, op)
	end
end

scalar_type(s::QMeasure) = Float64

# name(s::QMeasure) = "Q:Z$(s.key)"
# cname(s::QMeasure) = "C:Z$(s.key)"

function apply(s::QMeasure, qstate::AbstractVector)
	swap!(qstate, 1, s.key)
	probability, istate = _local_measure(qstate, s.op)
	ss = div(length(qstate), 2)
	r = (transpose(s.op[:, istate])*reshape(qstate, (2, ss)))/sqrt(probability) 
	swap!(qstate, 1, s.key)
	return reshape(r, length(r)), istate-1, probability	
end

function apply!(x::QMeasure, qstate::AbstractVector)
	swap!(qstate, 1, x.key)
	probability, istate = _local_measure(qstate, x.op)
	if x.auto_reset
		m = kron(x.op[:, istate], ZERO)/sqrt(probability)
	else
		m = kron(x.op[:, istate], x.op[:, istate])/sqrt(probability)
	end
	m = reshape(m, (2, 2))
	s = reshape(qstate, (1, 2, div(length(qstate), 2)))
	inplace_apply!(s, m)
	swap!(qstate, 1, x.key)	
	istate -= 1
	return istate, probability
end

measure(qstate::AbstractVector, pos::Int) = apply(QMeasure(pos), qstate)

measure!(qstate::AbstractVector, pos::Int; auto_reset::Bool=true) = apply!(QMeasure(pos, auto_reset=auto_reset), qstate)

# function measure(s::QMeasure, qstate::StateVector) 
# 	swap!(qstate, 1, s.key)
# 	probability, istate = _local_measure(qstate, s.op)
# 	ss = div(length(qstate), 2)
# 	if s.auto_reset
# 		r = (ZERO'*reshape(data(qstate), (2, ss)))/sqrt(probability) 
# 	else
# 		r = (transpose(s.op[:, istate])*reshape(data(qstate), (2, ss)))/sqrt(probability) 
# 	end
# 	swap!(qstate, 1, s.key)
# 	return StateVector(reshape(r, length(r))), istate-1, probability
# end
