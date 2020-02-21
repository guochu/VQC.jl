export generateprodham


function generateprodham(idens::Vector{<:AbstractMatrix}, opstr::AbstractDict)
	L = length(idens)
	vl = Vector{Any}(undef, L)
	for i in 1:L
	    v = get(opstr, i, nothing)
	    if v === nothing
	        vl[i] = idens[i]
	    else
	    	(size(v) == size(idens[i])) || error("wrong matrix size.")
	    	vl[i] = v
	    end
	end
	return sparse(kron(vl...))
end