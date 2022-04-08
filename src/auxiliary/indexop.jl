

"""	
	move_selected_index_forward(a, I)
	move the indexes specified by I to the front of a
	# Arguments
	@ a::NTuple{N, Int}: the input tensor.
	@ I: tuple or vector of integer.
"""
function move_selected_index_forward(a::Vector{T}, I) where {T}
    na = length(a)
    nI = length(I)
    b = Vector{T}(undef, na)
    k1 = 0
    k2 = nI
    for i=1:na
        s = 0
        while s != nI
        	if i == I[s+1]
        		b[s+1] = a[k1+1]
        	    k1 += 1
        	    break
        	end
        	s += 1
        end
        if s == nI
        	b[k2+1]=a[k1+1]
        	k1 += 1
            k2 += 1
        end
    end
    return b
end

function move_selected_index_forward(a::NTuple{N, T}, I) where {N, T}
    return NTuple{N, T}(move_selected_index_forward([a...], I))
end

"""	
	move_selected_index_backward(a, I)
	move the indexes specified by I to the back of a
	# Arguments
	@ a::NTuple{N, Int}: the input tensor.
	@ I: tuple or vector of integer.
"""
function move_selected_index_backward(a::Vector{T}, I) where {T}
	na = length(a)
	nI = length(I)
	nr = na - nI
	b = Vector{T}(undef, na)
	k1 = 0
	k2 = 0
	for i = 1:na
	    s = 0
	    while s != nI
	    	if i == I[s+1]
	    		b[nr+s+1] = a[k1+1]
	    		k1 += 1
	    		break
	    	end
	    	s += 1
	    end
	    if s == nI
	        b[k2+1] = a[k1+1]
	        k2 += 1
	        k1 += 1
	    end
	end
	return b
end

function move_selected_index_backward(a::NTuple{N, T}, I) where {N, T}
	return NTuple{N, T}(move_selected_index_backward([a...], I))
end
