export Chain


struct Chain
	data::Vector{<:AbstractDifferentiableQuantumOperation}
end

Chain() = Chain(Vector{AbstractDifferentiableQuantumOperation}())
Chain(items::AbstractDifferentiableQuantumOperation...) = Chain([items...])

@adjoint Chain(items::Vector{<:AbstractDifferentiableQuantumOperation}) = Chain(items), z -> (z,)

# @adjoint Chain(items::AbstractQuantumOperation...) = Chain(items...), z -> (z...,)

data(s::Chain) = s.data

Base.getindex(x::Chain, i::Int) = Base.getindex(data(x), i)
Base.setindex!(x::Chain, v::AbstractDifferentiableQuantumOperation,  i::Int) = Base.setindex!(data(x), v, i)
Base.length(x::Chain) = Base.length(data(x))
Base.iterate(x::Chain) = Base.iterate(data(x))
Base.iterate(x::Chain, state) = Base.iterate(data(x), state)
Base.eltype(x::Chain) = Base.eltype(data(x))

Base.isempty(x::Chain) = Base.isempty(data(x))
Base.empty!(x::Chain) = empty!(data(x))

add!(x::Chain, s) = push!(x, s)
Base.push!(x::Chain, s::AbstractDifferentiableQuantumOperation) = Base.push!(data(x), s)
Base.reverse(x::Chain) = Chain(reverse(data(x)))



function *(m::Chain, x::AbstractVector)
    xtmp = copy(x)
	for item in m
	    xtmp = item * xtmp
	end
	return xtmp
end


@adjoint *(m::Chain, x::AbstractVector) = begin
    y = m * x
    return y, z -> begin
        ytmp = copy(y)
        ztmp = conj(z)
        r = []
        for item in reverse(m)
            ytmp, grad, ztmp = backward_evolution(ytmp, item, ztmp)
            push!(r, grad)
        end
        return reverse(r), conj(ztmp)
    end
end

collect_variables_impl!(r::Vector, s::Chain) = collect_variables_impl!(r, data(s))
set_parameters_impl!(s::Chain, coeff::AbstractVector{<:Number}, start_pos::Int=1) = set_parameters_impl!(
    data(s), coeff, start_pos)
