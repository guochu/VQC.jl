


"""
	compute cumulant dim (cudim) from the dimension of
	the tensor
"""
function dim2cudim_col(dim)
	cudim = Vector{Int}(undef, length(dim))
	cudim[1] = 1
	temp = 1
	for i in 2:length(dim)
		temp *= dim[i-1]
		cudim[i] = temp
	end
	return cudim
end

"""
	singleindex is a single number, numtindex and cudim
	are list. singleindex and cudim are input, multindex
	are output. This function map the single index to
	multindex which is useful for general tensor index
	operation
"""
function sind2mind_col(singleindex::Int128, cudim)
	L = length(cudim)
	multindex = Vector{Int}(undef, L)
	for i in L:-1:2
		multindex[i] = div(singleindex, cudim[i])
		singleindex %= cudim[i]
	end
	multindex[1] = div(singleindex, cudim[1])
	return multindex
end

"""
	map multindex to singleindex
"""
function mind2sind_col(multindex, cudim)
	L = length(cudim)
	(length(multindex)==L) || error("wrong multindex size.")
	singleindex = Int128(multindex[1])
	for i in 2:L
		singleindex += multindex[i]*cudim[i]
	end
	return singleindex
end

sub2ind(d::Int, ind::Int) = begin
    (ind < d) || error("index out of range.")
    return ind
end

ind2sub(d::Int, ind::Int) = sub2ind(d, ind)

function sub2ind(dims::Union{Vector, Tuple}, multiindex::Union{Vector, Tuple})
	N = length(dims)
	(N == length(multiindex)) || error("size of mutliple index mismatch with ZipIndex.")
	(N >= 1) || error("no index.")
	ind = multiindex[N]
	for i = N-1:-1:1
		r1 = dims[i]
		(multiindex[i] < r1) || error("the $i-th index out of range.")
		ind = multiindex[i] + r1*ind
    end
    return ind
end

function ind2sub(dims::Union{Vector, Tuple}, ind::Integer)
	N = length(dims)
	(N >= 1) || error("no index.")
	multiindex = Vector{Int}(undef, N)
	indnext = 0
	for i in 1:N-1
	    r1 = dims[i]
	    indnext = div(ind, r1)
	    multiindex[i] = ind-r1*indnext
	    ind = indnext
	end
	(indnext < dims[N]) || error("$ind out of range.")
	multiindex[N] = indnext
	return multiindex
end
