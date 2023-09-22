
function swap!(qstate::AbstractVector, i::Int, j::Int)
	(i == j) && return 
	if i < j
		fsize = 2^(i-1)
		msize = 2^(j-i-1)
		bsize = div(length(qstate), (fsize*4*msize))
		s = reshape(qstate, (fsize, 2, msize, 2, bsize))

		# tmp = Tensor{T}(undef, fsize, msize, bsize)
		tmp = s[:, 1, :, 2, :]
		s[:, 1, :, 2, :] = s[:, 2, :, 1, :]
		s[:, 2, :, 1, :] = tmp
	else
		swap!(qstate, j, i)
	end
end

function _entropy(v::AbstractVector{<:Real}) 
    a = [(abs(item) <= 1.0e-12) ? 0. : item for item in v]
    a = [item for item in a if (item != 0.)]
    s = sum(a)
    a ./= s
    return -dot(a, log2.(a))
end

function entropy(v::AbstractArray{T, 1}) where {T <: Real}
	a = _check_and_filter(v)
	return -dot(a, log2.(a))
end

function renyi_entropy(v::AbstractArray{T, 1}; α::Real=1) where T
	α = convert(T, α)
	if α==one(α)
	    return entropy(v)
	else
		a = _check_and_filter(v)
		a = v.^(α)
		return (1/(1-α)) * log2(sum(a))
	end
end


function _check_and_filter(v::AbstractArray{<:Real}; tol::Real=1.0e-12)
	(abs(sum(v) - 1) <= tol) || throw(ArgumentError("sum of singular values not equal to 1"))
	oo = zero(eltype(v))
	tol = convert(eltype(v), tol)
	for item in v
		((item < oo) && (-item > tol)) && throw(ArgumentError("negative singular values"))
	end
	# return [(abs(item) <= tol) ? oo : item for item in v]
	return [item for item in v if abs(item) > tol] 
end