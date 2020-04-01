export generateprodham


function generateprodham(idens::Vector{<:AbstractMatrix}, opstr::AbstractDict)
	L = length(idens)
	vl = Vector{Any}(undef, L)
	for i in 1:L
	    v = get(opstr, i, nothing)
	    if v === nothing
	        vl[i] = sparse(idens[i])
	    else
	    	(size(v) == size(idens[i])) || error("wrong matrix size.")
	    	vl[i] = sparse(v)
	    end
	end
	return kron(vl...)
end