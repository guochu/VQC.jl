
function swap!(qstate::AbstractVector, i::Int, j::Int) where T
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