

import Base.*
using Zygote
using Zygote: @adjoint
using VQC
using SparseArrays
import VQC: collect_variables_impl!, set_parameters_impl!

function real_variational_circuit_1d(L::Int, depth::Int)
	circuit = QCircuit()
	for i in 1:L
		add!(circuit, RyGate(i, Variable(randn())))
	end
	for i in 1:depth
		for j in 1:(L-1)
		    add!(circuit, CNOTGate((j, j+1)))
		end
		for j in 1:L
			add!(circuit, RyGate(j, Variable(randn())))
		end
	end
	return circuit
end

struct QCNNLayer
	storage::Vector{QCircuit}
	filter_shape::Tuple{Int, Int}
	padding::Int

	function QCNNLayer(s::Vector{QCircuit}, filter_shape::Tuple{Int, Int}; padding::Int=0)
		new(s, filter_shape, padding)
	end
end

storage(s::QCNNLayer) = s.storage
get_filter_shape(s::QCNNLayer) = s.filter_shape
nfilters(s::QCNNLayer) = length(storage(s))

# @adjoint get_filter_shape(s::QCNNLayer) = get_filter_shape(s), z->(nothing,)

function add_padding(m::AbstractArray{T, 3}, padding::Int) where {T <: Real}
	s1 = size(m, 1)
	s2 = size(m, 2)
	m1 = zeros(T, s1 + 2*padding, s2 + 2*padding, size(m, 3))
	m1[(padding+1):(padding+s1), (padding+1):(padding+s2), :] = m
	return m1
end

@adjoint add_padding(m::AbstractArray{T, 3}, padding::Int) where {T <: Real} = begin
	s1 = size(m, 1)
	s2 = size(m, 2)
    return add_padding(m, padding), z -> (z[(padding+1):(padding+s1), (padding+1):(padding+s2), :], nothing)
end

function collect_variables_impl!(a::Vector, b::QCNNLayer)
	for v in storage(b)
	    collect_variables_impl!(a, v)
	end
end

set_parameters_impl!(a::QCNNLayer, coeff::AbstractVector{<:Number}, start_pos::Int=1) = begin
    for v in storage(a)
    	start_pos = set_parameters_impl!(v, coeff, start_pos)
    end
    return start_pos
end

_observer(L::Int) = sparse(reduce(kron, [Z for i in 1:L]))

function qcnn_single_filter(m::Array{<:Real, 3}, circuit::QCircuit, filter_shape::Tuple{Int, Int})
	n1, n2, n3 = size(m)
	s1, s2 = filter_shape
	(s1 <= n1 && s2 <= n2) || error("filter size too large.")
	out = Array{Float64, 3}(undef, n1-s1+1, n2-s2+1, n3)
	L = s1 * s2
	ob = _observer(L)
	state_a = qstate(Float64, L)
	for i in 1:(n1-s1+1)
		for j in 1:(n2-s2+1)
			for k in 1:n3
				# state_a = qstate(Float64, reshape(m[i:(i+s1-1), j:(j+s2-1), k], s1*s2))
				reset!(state_a, reshape(m[i:(i+s1-1), j:(j+s2-1), k], L))
				# out[i, j, k] = dot(state_b, circuit * state_a)^2
                # tmp_state = circuit * state_a
				apply!(circuit, state_a)
				out[i, j, k] = dot(state_a, ob, state_a)
			end
	    end
	end
	return out
end

qcnn_single_filter(m::Array{<:Real, 2}, circuit, filter_shape) = qcnn_single_filter(
	reshape(m, size(m,1), size(m,2), 1), circuit, filter_shape)

_expec(op, x) = real(dot(x, op, x))
@adjoint _expec(op, x) = _expec(op, x), z -> (nothing, (2 * real(z)) .* (op * x),)


# _rqstate(v) = qstate(Float64, v)
# function _single_filter(circuit, v, op)
#     wspace = _rqstate(reshape(v, length(v)))
# 	return _expec(op, circuit * wspace)
# end

# @adjoint _single_filter(op, circuit, v, wspace) = begin
# 	new_circuit = copy(circuit)
# 	append!(new_circuit, [RyGate(i, Variable(theta*pi)) for (i, theta) in enumerate(v)])
# 	reset!(wspace)
# 	apply!(new_circuit, wspace)
# 	r, f = Zygote.pullback(_expec, op, wspace)
# 	return r, z -> begin
# 		zt = f(z)[2]
# 		tmp, grad, zt = backward_evolution(wspace, new_circuit, zt)
# 		# L = length(grad)
# 		n = length(v)
# 		return nothing, grad[1:n], grad[n:end] .* pi, nothing
# 	end
# end

@adjoint qcnn_single_filter(m::Array{<:Real, 3}, circuit::QCircuit, filter_shape::Tuple{Int, Int}) = begin
	n1, n2, n3 = size(m)
	s1, s2 = filter_shape
	(s1 <= n1 && s2 <= n2) || error("filter size too large.")
	out = Array{Float64, 3}(undef, n1-s1+1, n2-s2+1, n3)
	dout = Array{Any, 3}(undef, n1-s1+1, n2-s2+1, n3)
	L = s1 * s2
	ob = _observer(L)
	# f(c, a) = begin
    #     tmp_state = c * qstate(Float64, reshape(a, L))
    #     return _expec(ob, tmp_state)
    # end
	f(c, a) = _expec(ob, c * qstate(Float64, reshape(a, length(a))) )
	for i in 1:(n1-s1+1)
		for j in 1:(n2-s2+1)
			for k in 1:n3
				# state_a = qstate(reshape(m[i:(i+s1-1), j:(j+s2-1), k], s1*s2))
				# tmp, dtmp =   dot(state_b, circuit * state_a)
				tmp, dtmp = Zygote.pullback(f, circuit, m[i:(i+s1-1), j:(j+s2-1), k])
				out[i, j, k] = tmp
				dout[i, j, k] = dtmp
			end
	    end
	end
	return out, z -> begin
		dm = zeros(size(m))
		da = zeros(nparameters(circuit))
		for i in 1:(n1-s1+1)
			for j in 1:(n2-s2+1)
				for k in 1:n3
					a, b = dout[i, j, k](z[i, j, k])
					da .+= a
					dm[i:(i+s1-1), j:(j+s2-1), k] .+= b
				end
	    	end
		end
		return dm, da, nothing
	end
end

function _qcnn_impl(circuit::QCNNLayer, m::Array{<:Real, 3})
	# (circuit.padding != 0) && (m = add_padding(m, circuit.padding))
	r = Vector{Any}(undef, nfilters(circuit))
	for i in 1:nfilters(circuit)
	    r[i] = qcnn_single_filter(m, storage(circuit)[i], get_filter_shape(circuit))
	end
	isempty(r) && error("QCNN is empty.")
	out = Array{Float64, 4}(undef, size(r[1])..., length(r))
	for i in 1:length(r)
	    out[:,:,:, i] = r[i]
	end
	return reshape(out, size(out,1), size(out,2), size(out,3)*size(out,4))
end


@adjoint _qcnn_impl(circuit::QCNNLayer, m::Array{<:Real, 3}) = begin
	# (circuit.padding != 0) && (m = add_padding(m, circuit.padding))
	r = Vector{Any}(undef, nfilters(circuit))
	dr = Vector{Any}(undef, nfilters(circuit))
	for i in 1:nfilters(circuit)
	    r[i], dr[i] = Zygote.pullback(qcnn_single_filter, m, storage(circuit)[i],  get_filter_shape(circuit))
	end
	isempty(r) && error("QCNN is empty.")
	out = Array{Float64, 4}(undef, size(r[1])..., length(r))
	for i in 1:length(r)
	    out[:,:,:, i] = r[i]
	end
	return reshape(out, size(out,1), size(out,2), size(out,3)*size(out,4)), z -> begin
		z1 = reshape(z, size(out)...)
	    dm = zeros(Float64, size(m)...)
	    dc =  Vector{Vector{Float64}}(undef, nfilters(circuit))
		for i in 1:nfilters(circuit)
	    	a, b, dum = dr[i](z1[:,:,:,i])
	    	dm .+= a
	    	dc[i] = b
		end
		return dc, dm
	end
end

qcnn(circuit::QCNNLayer, m::Array{<:Real, 3}) = _qcnn_impl(circuit, add_padding(m, circuit.padding))


qcnn(circuit::QCNNLayer, m::Array{<:Real, 2}) = qcnn(circuit, reshape(m, size(m,1), size(m,2), 1))
*(circuit::QCNNLayer, m::Array) = qcnn(circuit, m)
