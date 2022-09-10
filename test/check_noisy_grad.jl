

function noisy_circuit(L, depth; kwargs...)
	circuit = variational_circuit_1d(L, depth; kwargs...)
	for i in 1:L
		push!(circuit, Depolarizing(i, p=0.1))
	end
	return circuit
end

function check_noisy_circuit_grad(L, depth)
	circuit = noisy_circuit(L, depth)
	initial_state = DensityMatrix(ComplexF64, L)

	loss(x) = real( sum(storage(x * initial_state)) )
	loss_fd(θs) = loss(noisy_circuit(L, depth, θs=θs))

	grad1 = gradient(loss, circuit)[1]
	grad2 = fdm_gradient(loss_fd, active_parameters(circuit))
	return maximum(abs.(grad1 - grad2)) < 1.0e-6
end

function check_noisy_circuit_expec_grad(L, depth)
	circuit = noisy_circuit(L, depth)
	m = QubitsTerm(1=>"+", L-1=>"-", coeff=0.7) + QubitsTerm(1=>"-", L-1=>"+", coeff=0.77) + QubitsTerm(1=>"X", L=>"Y", coeff=0.36)
	initial_state = DensityMatrix(ComplexF64, L)

	loss(x) = real(expectation(m, x * initial_state))
	loss_fd(θs) = loss(noisy_circuit(L, depth, θs=θs))

	grad1 = gradient(loss, circuit)[1]
	grad2 = fdm_gradient(loss_fd, active_parameters(circuit))
	# println("error is $(maximum(abs.(grad1 - grad2)))")
	return maximum(abs.(grad1 - grad2)) < 1.0e-6
end

function check_ham_term_grad(::Type{T}, L) where T
	state = rand_densitymatrix(T, L)

	m = QubitsTerm(1=>"+", L-1=>"Y", coeff=0.36)

	loss(x) = abs(expectation(m, x))
	loss_fd(θs) = loss(DensityMatrix(θs))

	grad1 = gradient(loss, state)[1]
	grad2 = fdm_gradient(loss_fd, storage(state))
	return maximum(abs.(grad1 - grad2)) < 1.0e-6
end



@testset "gradient of noisy quantum circuit" begin
	for L in 2:5
		@test check_ham_term_grad(ComplexF64, L)
		for depth in 0:5
		    @test check_noisy_circuit_grad(L, depth)
		    @test check_noisy_circuit_expec_grad(L, depth)
		end	    
	end
end