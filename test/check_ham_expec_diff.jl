
function check_ham_term_grad(::Type{T}, L) where T
	state = rand_state(T, L)

	m = QubitsTerm(1=>"+", L-1=>"Y", coeff=0.36)
	loss(x) = abs(expectation(m, x))
	loss_fd(θs) = loss(StateVector(θs))

	grad1 = gradient(loss, state)[1]
	grad2 = fdm_gradient(loss_fd, amplitudes(state))
	return maximum(abs.(grad1 - grad2)) < 1.0e-6
end

function check_ham_expec_grad(::Type{T}, L) where T
	state = rand_state(T, L)

	m = QubitsTerm(1=>"+", L-1=>"-", coeff=0.77) + QubitsTerm(1=>"-", L-1=>"+", coeff=0.77) + QubitsTerm(1=>"X", L=>"Y", coeff=0.36)
	loss(x) = abs(expectation(m, x))
	loss_fd(θs) = loss(StateVector(θs))

	grad1 = gradient(loss, state)[1]
	grad2 = fdm_gradient(loss_fd, amplitudes(state))
	return maximum(abs.(grad1 - grad2)) < 1.0e-6
end

function check_qterm_expec_grad_long(::Type{T}, L, n) where T
	state = rand_state(T, L)

	m = QubitsTerm(Dict(i=>"X" for i in 1:n), coeff=0.77)
	loss(x) = abs(expectation(m, x))
	loss_fd(θs) = loss(StateVector(θs))

	grad1 = gradient(loss, state)[1]
	grad2 = fdm_gradient(loss_fd, amplitudes(state))
	return maximum(abs.(grad1 - grad2)) < 1.0e-6
end

function check_ham_expec_grad_long(::Type{T}, L, n) where T
	state = rand_state(T, L)

	m = QubitsTerm(Dict(i=>"X" for i in 1:n), coeff=0.77) + QubitsTerm(1=>"X", coeff=1.2) + QubitsTerm(1=>"X", 2=>"Z", coeff=0.3)
	loss(x) = abs(expectation(m, x))
	loss_fd(θs) = loss(StateVector(θs))

	grad1 = gradient(loss, state)[1]
	grad2 = fdm_gradient(loss_fd, amplitudes(state))
	return maximum(abs.(grad1 - grad2)) < 1.0e-6
end

function check_mps_ham_expec_grad(::Type{T}, L::Int) where {T<:Number}
	observer = QubitsTerm(1=>"Z", 3=>"Z")
	loss(v) = real(expectation(observer, qubit_encoding(T, v)))

	v = randn(L)

	grad1 = gradient(loss, v)[1]
	grad2 = fdm_gradient(loss, v)
	return maximum(abs.(grad1 - grad2)) < 1.0e-6	
end


@testset "gradient of quantum operator expectation value" begin
	for L in 3:6
		@test check_ham_term_grad(ComplexF64, L)
		@test check_ham_expec_grad(ComplexF64, L)
		for n in 1:L
		    @test check_qterm_expec_grad_long(ComplexF64, L, n)
		    @test check_ham_expec_grad_long(ComplexF64, L, n)
		end
	end
    for T in [Float64, ComplexF64]
    	for L in 3:6
			@test check_mps_ham_expec_grad(ComplexF64, L)
		end 
    end
end

