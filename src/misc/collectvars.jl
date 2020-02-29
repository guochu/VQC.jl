export collect_variables, parameters

collect_variables_impl!(a::Vector, b::Nothing) = nothing
collect_variables_impl!(a::Vector, b::Number) = push!(a, b)
function collect_variables_impl!(a::Vector, b::AbstractArray) 
	for item in b
	    collect_variables_impl!(a, item)
	end
end

function collect_variables_impl!(a::Vector, b::AbstractDict)
	for (k, v) in b
	    collect_variables_impl!(a, v)
	end
end

function collect_variables_impl!(a::Vector, b::NamedTuple)
	for v in b
	    collect_variables_impl!(a, v)
	end
end

function collect_variables_impl!(a::Vector, b::Tuple)
	for v in b
	    collect_variables_impl!(a, v)
	end
end

"""
    collect_variables(args...)
Collect variables from args...
"""
function collect_variables(args...)
	a = Number[]
	for item in args
	    collect_variables_impl!(a, item)
	end
	return a
end

"""
    parameters(args...) = collect_variables(args...)
"""
parameters(args...) = collect_variables(args...)
