export set_parameters!

function set_parameters_impl!(a::AbstractVector, coeff::AbstractVector{<:Number}, start_pos::Int) 
    for j in 1:length(a)
        if isa(a[j], Number)
            a[j] = coeff[start_pos]
            start_pos += 1
        else
            start_pos = set_parameters_impl!(a[j], coeff, start_pos)
        end
    end
    return start_pos
end

set_parameters_impl!(a::AbstractArray{<:Number, N}, coeff::AbstractVector{<:Number}, start_pos::Int=1) where N = begin
    for j in 1:length(a)
        a[j] = coeff[start_pos] 
        start_pos += 1
    end
    return start_pos
end

set_parameters_impl!(a::AbstractArray{T, N}, coeff::AbstractVector{<:Number}, start_pos::Int=1) where {T, N} = begin
    for j in 1:length(a)
        start_pos = set_parameters_impl!(a[j], coeff, start_pos)
    end
    return start_pos   
end

function set_parameters_impl!(a::Tuple, coeff::AbstractVector{<:Number}, start_pos::Int=1)
    for item in a
        start_pos = set_parameters_impl!(item, coeff, start_pos)
    end
    return start_pos
end

"""
    set_parameters!(coeff::AbstractVector{<:Number}, args...)
Set parameters for args...
"""
function set_parameters!(coeff::AbstractVector{<:Number}, args...)
    start_pos = 1
    for item in args
        start_pos = set_parameters_impl!(item, coeff, start_pos)
    end
    (start_pos == length(coeff)+1) || @warn "only $(start_pos-1) out of $(length(coeff)) parameters has been used."
end