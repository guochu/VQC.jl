
function check_measure_1()
	state = rand_state(Float64, 3)
	i, p = measure!(state, 1, auto_reset=true)
	tmp, i1, p1 = measure(state, 1)
	return i1 == 0 && isapprox(p1, 1, atol=1.0e-10)
end

function check_measure_2()
	state = rand_state(ComplexF64, 3)
	i, p = measure!(state, 2, auto_reset=false)
	tmp, i1, p1 = measure(state, 2)
	return i1 == i && isapprox(p1, 1, atol=1.0e-10)
end

function check_select_1()
	state = rand_state(Float64, 3)
	post_select!(state, 2, 0)
	tmp, i1, p1 = measure(state, 2)
	return i1 == 0 && isapprox(p1, 1, atol=1.0e-10)
end

function check_select_2()
	state = rand_state(Float64, 3)
	post_select!(state, 3, 1)
	tmp, i1, p1 = measure(state, 3)
	return i1 == 1 && isapprox(p1, 1, atol=1.0e-10)
end

function check_select_3()
	state = (sqrt(2)/2) * (qubit_encoding([0, 0]) + qubit_encoding([1, 1]))
	r, p = post_select(state, 1, 0)
	return isapprox(storage(r), Gates.ZERO)
end

function check_select_4()
	state = (sqrt(2)/2) * (qubit_encoding([0, 0]) + qubit_encoding([1, 1]))
	r, p = post_select(state, 1, 1)
	return isapprox(storage(r), Gates.ONE)
end

function check_select_grad()
	initial_state = qubit_encoding([0, 1])
	function loss(s) 
		r, p = post_select(s, 2, 1)
		return distance(r, initial_state) + p^2
	end
	loss_fd(θs) = loss(StateVector(θs))

	x = rand_state(ComplexF64, 3)
	grad1 = gradient(loss, x)[1]
	grad2 = fdm_gradient(loss_fd, storage(x))
	return maximum(abs.(grad1 - grad2)) < 1.0e-6
end

@testset "check quantum measure" begin
	@test check_measure_1()
	@test check_measure_2()
end

@testset "check quantum select" begin
	@test check_select_1()
	@test check_select_2()
	@test check_select_3()
	@test check_select_4()
end

@testset "check quantum select grad" begin
	@test check_select_grad()
end





