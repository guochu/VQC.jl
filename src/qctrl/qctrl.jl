
export ctrlham, evolve

struct ControlHamiltonian{A, B} <: AbstractHamiltonianExponential
	Hi::A
	Hc::B
	fvals::Vector{Float64}
	dt::Float64
end

ctrlham(a::AbstractMatrix, b::AbstractMatrix, fvals::Vector{Float64}; dt::Real=0.1) = ControlHamiltonian(
	a, b, copy(fvals), dt)

ctrlham(a::Hamiltonian, b::Hamiltonian, fvals::Vector{Float64}; dt::Real=0.1) = ctrlham(matrix(a), matrix(b), fvals; dt=dt)


get_hi(s::ControlHamiltonian) = s.Hi
get_hc(s::ControlHamiltonian) = s.Hc

get_fvals(s::ControlHamiltonian) = s.fvals
get_dt(s::ControlHamiltonian) = s.dt

num_steps(s::ControlHamiltonian) = length(get_fvals(s))

Base.size(s::ControlHamiltonian) = size(get_hi(s))
Base.size(s::ControlHamiltonian, j::Int) = size(get_hi(s), j)

function get_mat(m::ControlHamiltonian, j::Int)
	v = get_fvals(m)[j]
	return get_hi(m) + v * get_hc(m)
end


# function evolve_one_step(m::ControlHamiltonian, j::Int, x::AbstractVector, rev::Bool=false)

# 	mat = get_mat(m, j)
# 	# if rev
# 	# 	y, info = exponentiate(mat, im * dt, x; tol=EXPM_TOL, maxiter=EXPM_MAXITER, ishermitian=true)
# 	# else
# 	# 	y, info = exponentiate(mat, -im * dt, x; tol=EXPM_TOL, maxiter=EXPM_MAXITER, ishermitian=true)
# 	# end
	
# 	# (info.converged==1) || error("expm fail to converge.")

# 	if rev
# 		y = exp(mat .* (im * dt)) * x
# 	else
# 		y = exp(mat .* (-im * dt)) * x
# 	end
	
# 	return y
# end

evolve_mat(mat::AbstractMatrix, dt::Number, x::AbstractVector) = begin
	try
		y, info = exponentiate(mat, dt, x; tol=EXPM_TOL, maxiter=EXPM_MAXITER, ishermitian=true)
		(info.converged==1) || error("expm fail to converge.")
		return y
	catch
		return exp(Matrix(mat) .* dt) * x
	end
end 

function compute_diff_expec(m::ControlHamiltonian, j::Int, vo::AbstractVector, vi::AbstractVector)
	v = get_fvals(m)[j]
	dt = get_dt(m)
	step_size = QCTRL_DIFF_STEP_SIZE
	h1 = get_hi(m) + v * get_hc(m)
	h2 = h1 + step_size * get_hc(m)
	h = vdot(vo, evolve_mat(h2, -im*dt, vi) - evolve_mat(h1, -im*dt, vi))
	return h / step_size 
end


vdot(a::AbstractArray, b::AbstractArray) = dot(conj(a), b)

*(m::ControlHamiltonian, x::AbstractVector) = evolve(m, x)

function evolve(m::ControlHamiltonian, x::AbstractVector)
	for j in 1:num_steps(m)
	    x = evolve_mat(get_mat(m, j), -im*get_dt(m), x)
	end
	return x
end

function backward_evolution(y::AbstractVector, m::ControlHamiltonian, z::AbstractVector)
	r = []
	ytmp = copy(y)
	zt = copy(z)

	for j in num_steps(m):-1:1
		# println("j is $j.")
		dt = get_dt(m)

		h = get_mat(m, j)

		ytmp = evolve_mat(h, im * dt, ytmp)

        # println("mat is $((-im * dt) * get_hc(m)).")
        # df =  (-im * dt) * vdot(zt, get_hc(m) * ytmp)

        df = compute_diff_expec(m, j, zt, ytmp)
        # println("df is $df.")
        push!(r, real(df))

        zt = evolve_mat(transpose(h), -im * dt, zt)

        # println("dot result is $(vdot(zt, ytmp))---------------")

    end	
    # println("difference is $(ytmp - x)")
    return ytmp, reverse([r...]), zt
end

@adjoint evolve(m::ControlHamiltonian, x::AbstractVector) = begin
    y = evolve(m, x)
    return y, z -> begin
        ytmp, grad, zt = backward_evolution(y, m, conj(z))
        return grad, conj(zt)
    end 
end



collect_variables_impl!(r::Vector, s::ControlHamiltonian) = collect_variables_impl!(r, get_fvals(s))
set_parameters_impl!(s::ControlHamiltonian, coeff::AbstractVector{<:Number}, start_pos::Int=1) = set_parameters_impl!(
    get_fvals(s), coeff, start_pos)

