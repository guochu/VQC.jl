export HamExp, expham, get_time


struct HamExp{M, T} <: AbstractHamiltonianExponential
	h::M
	dt::T
end


get_h(s::HamExp) = s.h
get_dt(s::HamExp) = s.dt
get_time(s::HamExp) = value(get_dt(s))


Base.transpose(m::HamExp) = HamExp(transpose(get_h(m)), get_dt(m))
Base.adjoint(m::HamExp) = HamExp(get_h(m), -get_dt(m))
Base.size(s::HamExp) = size(get_h(s))
Base.size(s::HamExp, j::Int) = size(get_h(s), j)


*(m::HamExp, v::AbstractVector) = evolve_mat(get_h(m), -im*value(get_dt(m)), v)
*(v::AbstractVector, m::HamExp) = transpose(m) * v


expham(m::AbstractMatrix, t::Real) = HamExp(m, t)
expham(m::AbstractMatrix, t::Variable{<:Real}) = HamExp(m, t)

expham(m::Hamiltonian, t) = expham(matrix(m), t)

apply(m::Hamiltonian, t::Real, v::AbstractVector) = expham(m, t) * v