

expham(h::AbstractMatrix) = expham(h, Variable(rand()))

expham(h::Hamiltonian) = expham(matrix(h))


backward_evolution(y::AbstractVector, m::HamExp{M, <:Variable}, z::AbstractVector) where M = (
	expham(get_h(m), -value(get_dt(m))) * y, real(-im*vdot(z, get_h(m) * y)), z * m)

# @adjoint *(m::HamExp, x::AbstractVector) = begin
#     y = m * x
#     return y, z -> (real(-im*vdot(conj(z), get_h(m) * y)), conj(evolve(transpose(m), conj(z))) )
# end

@adjoint *(m::HamExp{M, <:Variable}, x::AbstractVector) where M = begin
    y = m * x
    return y, z -> (real(-im*vdot(conj(z), get_h(m) * y)), m' * z)
end

@adjoint *(m::HamExp{M, <:Real}, x::AbstractVector) where M = begin
    y = m * x
    return y, z -> (nothing, m' * z)
end


collect_variables_impl!(r::Vector, s::HamExp{M, <:Variable}) where M = collect_variables_impl!(r, get_dt(s))
collect_variables_impl!(r::Vector, s::HamExp{M, <:Real}) where M = nothing
set_parameters_impl!(s::HamExp{M, <:Variable}, coeff::AbstractVector{<:Number}, start_pos::Int=1) where M = set_parameters_impl!(
    get_dt(s), coeff, start_pos)
set_parameters_impl!(s::HamExp{M, <:Real}, coeff::AbstractVector{<:Number}, start_pos::Int=1) where M = start_pos


@adjoint get_time(s::HamExp{M, <:Variable}) where M = get_time(s), z -> (z,)
@adjoint get_time(s::HamExp{M, <:Real}) where M = get_time(s), z -> (nothing,)

