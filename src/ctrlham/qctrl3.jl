

struct ControlHamiltonian3{A, B} <: AbstractHamiltonianExponential
	Hi::A
	Hc::Vector{B}
	fvals::Vector{Vector{Float64}}
	times::Vector{Float64}
end

ctrlham(a::AbstractMatrix, b::Vector{<:AbstractMatrix}, nparas::Int) = ControlHamiltonian3(a, b, 
	[rand(nparas) for i in 1:length(b)], rand(nparas))


get_hi(s::ControlHamiltonian3) = s.Hi
get_hcs(s::ControlHamiltonian3) = s.Hc

get_fvals(s::ControlHamiltonian3) = s.fvals
get_times(s::ControlHamiltonian3) = s.times

# this function may be used in the loss function
times(s::ControlHamiltonian3) = get_times(s)

num_steps(s::ControlHamiltonian3) = length(get_times(s))

num_hcs(s::ControlHamiltonian3) = length(get_hcs(s))


Base.size(s::ControlHamiltonian3) = size(get_hi(s))
Base.size(s::ControlHamiltonian3, j::Int) = size(get_hi(s), j)


function get_mat(m::ControlHamiltonian3, j::Int)	
	h = get_hi(m)
	for i in 1:num_hcs(m)
	    h += get_fvals(m)[i][j] * get_hcs(m)[i]
	end
	return return h
end

function compute_diff_expec(m::ControlHamiltonian3, j::Int, vo::AbstractVector, vi::AbstractVector)
	r = []
	dt = get_times(m)[j]
	step_size = QCTRL_DIFF_STEP_SIZE
	h1 = get_mat(m, j)
	for i in 1:num_hcs(m)
		h2 = h1 + step_size * get_hcs(m)[i]
	    hj = vdot(vo, evolve_mat(h2, -im*dt, vi) - evolve_mat(h1, -im*dt, vi))
	    push!(r, hj / step_size)
	end
	return [r...]
end

*(m::ControlHamiltonian3, x::AbstractVector) = evolve(m, x)

function evolve(m::ControlHamiltonian3, x::AbstractVector)
	for j in 1:num_steps(m)
	    x = evolve_mat(get_mat(m, j), -im*get_times(m)[j], x)
	end
	return x
end

function backward_evolution(y::AbstractVector, m::ControlHamiltonian3, z::AbstractVector)
	r1 = []
	r2 = []
	ytmp = copy(y)
	zt = copy(z)

	for j in num_steps(m):-1:1
		# println("j is $j.")
		dt = get_times(m)[j]

		h = get_mat(m, j)

		df = -im*vdot(zt, h * ytmp)

		push!(r2, real(df))

		ytmp2 = copy(ytmp)

		ytmp = evolve_mat(h, im * dt, ytmp)

        # println("mat is $((-im * dt) * get_hc(m)).")
        # df =  (-im * dt) * vdot(zt, get_hc(m) * ytmp)
        r = []
		for i in 1:num_hcs(m)
			h2 = h + QCTRL_DIFF_STEP_SIZE * get_hcs(m)[i]
	   	 	hj = vdot(zt, evolve_mat(h2, -im*dt, ytmp) - ytmp2)
	    	push!(r, hj / QCTRL_DIFF_STEP_SIZE)
		end

        # df = compute_diff_expec(m, j, zt, ytmp)
        # println("df is $df.")
        push!(r1, real([r...]))

        zt = evolve_mat(transpose(h), -im * dt, zt)

        # println("dot result is $(vdot(zt, ytmp))---------------")

    end	
    # println("difference is $(ytmp - x)")
    r1 = reverse([r1...])
    n1 = length(r1)
    n2 = length(r1[1])
    r3 = [[r1[i][j] for i in 1:n1] for j in 1:n2]
    r2 = reverse([r2...])
    return ytmp, collect_variables(r3, r2), zt
end


@adjoint evolve(m::ControlHamiltonian3, x::AbstractVector) = begin
    y = evolve(m, x)
    return y, z -> begin
        ytmp, grad, zt = backward_evolution(y, m, Vector{scalar_type(y)}(conj(z)) )
        return grad, conj(zt)
    end 
end

@adjoint get_times(m::ControlHamiltonian3) = get_times(m), z -> begin
    n = num_steps(m)
    (length(z) == n) || error("wrong input length.")
    L = (num_hcs(m)+1)*n
    r = zeros(eltype(z), L)
    r[(L-n+1):L] = z
    return (r,)
end



collect_variables_impl!(r::Vector, s::ControlHamiltonian3) = begin
	collect_variables_impl!(r, get_fvals(s))
	collect_variables_impl!(r, get_times(s))
end 
set_parameters_impl!(s::ControlHamiltonian3, coeff::AbstractVector{<:Number}, start_pos::Int=1) = begin
    start_pos = set_parameters_impl!(get_fvals(s), coeff, start_pos)
    return set_parameters_impl!(get_times(s), coeff, start_pos)
end 






