
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
	return L
end