
"""
	state gradient with distance loss function
"""
function state_grad_dot_real(L::Int, depth::Int)
	target_state = rand_state(ComplexF64, L)
	initial_state = rand_state(ComplexF64, L)
	circuit =  variational_circuit_1d(L, depth)

	loss(x) = real(dot(target_state, circuit * x))
	loss_fd(θs) = loss(StateVector(θs))

	grad1 = gradient(loss, initial_state)[1]
	grad2 = fdm_gradient(loss_fd, amplitudes(initial_state))
	return maximum(abs.(grad1 - grad2)) < 1.0e-6
end

"""
	state gradient with distance loss function
"""
function state_grad_dot_imag(L::Int, depth::Int)
	target_state = rand_state(ComplexF64, L)
	initial_state = rand_state(ComplexF64, L)
	circuit =  variational_circuit_1d(L, depth)

	loss(x) = imag(dot(target_state, circuit * x))
	loss_fd(θs) = loss(StateVector(θs))

	grad1 = gradient(loss, initial_state)[1]
	grad2 = fdm_gradient(loss_fd, amplitudes(initial_state))
	return maximum(abs.(grad1 - grad2)) < 1.0e-6
end

"""
	state gradient with distance loss function
"""
function state_grad_dot_abs(L::Int, depth::Int)
	target_state = rand_state(ComplexF64, L)
	initial_state = rand_state(ComplexF64, L)
	circuit =  variational_circuit_1d(L, depth)

	loss(x) = abs(dot(target_state, circuit * x))
	loss_fd(θs) = loss(StateVector(θs))

	grad1 = gradient(loss, initial_state)[1]
	grad2 = fdm_gradient(loss_fd, amplitudes(initial_state))
	return maximum(abs.(grad1 - grad2)) < 1.0e-6
end

"""
	state gradient with distance loss function
"""
function state_grad_distance(L::Int, depth::Int)
	target_state = rand_state(ComplexF64, L)
	initial_state = rand_state(ComplexF64, L)
	circuit =  variational_circuit_1d(L, depth)

	loss(x) = distance(target_state, circuit * x)
	loss_fd(θs) = loss(StateVector(θs))

	grad1 = gradient(loss, initial_state)[1]
	grad2 = fdm_gradient(loss_fd, amplitudes(initial_state))
	return maximum(abs.(grad1 - grad2)) < 1.0e-6
end


function qstate_grad(::Type{T}, L::Int) where {T<:Number}
	target_state = rand_state(T, L)
	loss(v) = abs(dot(target_state, qubit_encoding(T, v)))
	x = randn(L)

	grad1 = gradient(loss, x)[1]
	grad2 = fdm_gradient(loss, x)
	return maximum(abs.(grad1 - grad2)) < 1.0e-6
end

@testset "gradient of quantum state with loss function real(dot(a, circuit*x))" begin
	for L in 2:5
		for depth in 0:5
		    @test state_grad_dot_real(L, depth)
		end
	end
end

@testset "gradient of quantum state with loss function imag(dot(a, circuit*x))" begin
	for L in 2:5
		for depth in 0:5
		    @test state_grad_dot_imag(L, depth)
		end
	end
end

@testset "gradient of quantum state with loss function abs(dot(a, circuit*x))" begin
	for L in 2:5
		for depth in 0:5
		    @test state_grad_dot_abs(L, depth)
		end
	end
end

@testset "gradient of quantum state with loss function distance(a, circuit*x)" begin
	for L in 2:5
		for depth in 0:5
		    @test state_grad_distance(L, depth)
		end
	end
end


@testset "gradient of quantum state initializer qstate" begin
	for L in 2:5
		for T in [Float64, Complex{Float64}]
		    @test qstate_grad(T, L)
		end
	end
end
