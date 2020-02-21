export Variable

mutable struct Variable{T}
	value::T

	Variable(m::Number) = new{typeof(m)}(m)
end

+(a::Variable, b::Float64) = a.value + b
-(a::Variable, b::Float64) = a.value - b
-(a::Variable) = Variable(-value(a))
+(a::Variable) = a

value(x::Variable) = x.value
value(x) = x


collect_variables_impl!(a::Vector, b::Variable) = push!(a, value(b))
set_parameters_impl!(s::Variable, coeff::AbstractVector{<:Number}, start_pos::Int=1) = begin
    s.value = coeff[start_pos]
    return start_pos + 1
end