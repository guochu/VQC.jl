

@adjoint storage(x::StateVector) = storage(x), z -> (z,)
@adjoint StateVector(data::AbstractVector{<:Number}, n::Int) = StateVector(data, n), z -> (z, nothing)
@adjoint StateVector(data::AbstractVector{<:Number}) = StateVector(data), z -> (z,)

# @adjoint dot(x::StateVector, y::StateVector) = begin
# 	v, back = Zygote.pullback(dot, storage(x), storage(y))
# 	return v, z -> begin
# 		a, b = back(z)
# 		return StateVector(a, nqubits(x)), StateVector(b, nqubits(y))
# 	end
# end

# this is stupid, why should I need it
@adjoint dot(x::StateVector, y::StateVector) = Zygote.pullback(dot, storage(x), storage(y))

