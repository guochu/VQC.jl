export expham


struct HamExp{M} <: AbstractHamiltonianExponential
	h::M
	dt::Variable{Float64}
end

expham(h::AbstractMatrix, t::Float64) = HamExp{typeof(h)}(h, Variable(t))

expham(h::AbstractMatrix) = expham(h, rand())

expham(h::Hamiltonian) = expham(matrix(h))

Base.transpose(m::HamExp) = HamExp(transpose(get_h(m)), get_dt(m))

get_h(s::HamExp) = s.h
get_dt(s::HamExp) = s.dt


Base.size(s::HamExp) = size(get_h(s))
Base.size(s::HamExp, j::Int) = size(get_h(s), j)

evolve(m::HamExp, v::AbstractVector) = evolve_mat(get_h(m), -im*value(get_dt(m)), v)


backward_evolution(y::AbstractVector, m::HamExp, z::AbstractVector) = (evolve(expham(get_h(m), -value(get_dt(m))), y), 
	real(-im*vdot(z, get_h(m) * y)), evolve(transpose(m), z))


@adjoint evolve(m::HamExp, x::AbstractVector) = begin
    y = evolve(m, x)
    return y, z -> (real(-im*vdot(conj(z), get_h(m) * y)), conj(evolve(transpose(m), conj(z))) )
end

*(m::HamExp, v::AbstractVector) = evolve(m, v)


collect_variables_impl!(r::Vector, s::HamExp) = collect_variables_impl!(r, get_dt(s))
set_parameters_impl!(s::HamExp, coeff::AbstractVector{<:Number}, start_pos::Int=1) = set_parameters_impl!(
    get_dt(s), coeff, start_pos)



get_times(s::HamExp) = value(get_dt(s))

@adjoint get_times(s::HamExp) = get_times(s), z -> (z,)

times(m::HamExp) = get_times(m)