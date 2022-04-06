

"""
	circuit gradient with dot loss function
"""
function circuit2d_grad_dot_real(m::Int, n::Int, depth::Int)
	L = m * n
	target_state = rand_state(ComplexF64, L)
	initial_state = StateVector(ComplexF64, L)
	circuit =  variational_circuit_2d(m, n, depth)

	loss(x) = real(dot(target_state, x * initial_state))
	loss_fd(θs) = loss(variational_circuit_2d(m, n, depth, θs))

	grad1 = gradient(loss, circuit)[1]
	grad2 = fdm_gradient(loss_fd, active_parameters(circuit))
	return maximum(abs.(grad1 - grad2)) < 1.0e-6
end

@testset "gradient of 2d quantum circuit with loss function real(dot(x, circuit*y))" begin
	for depth in 0:5
		@test circuit2d_grad_dot_real(3, 4, depth)
	end	    
end