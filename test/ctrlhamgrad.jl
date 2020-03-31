using VQC: QCircuit, ctrlham, qrandn, qstate, check_gradient, RxGate, expham, get_times, Variable

using LinearAlgebra: dot

using Zygote

function build_rotational(L)
	circuit = QCircuit()
	for i in 1:L
	    push!(circuit, RxGate(i, Variable(rand()*2*pi)))
	end
	return circuit
end

function rand_herm_mat(n)
	h = randn(Complex{Float64}, 2^n, 2^n)
	return h' * h
end

function ctrlham_grad_dot_real(L::Int, nparas::Int)
	target_state = qrandn(Complex{Float64}, L)
	initial_state = qstate(Complex{Float64}, L)
	hi = rand_herm_mat(L)
	hc = rand_herm_mat(L)

	m = ctrlham(hi, hc, rand(nparas), dt=0.1)

	loss(x) = real(dot(target_state, x * initial_state))

	return check_gradient(loss, m, dt=1.0e-8, atol=1.0e-4)
end

function ctrlham_with_rx_grad_dot_real(L::Int, nparas::Int)
	target_state = qrandn(Complex{Float64}, L)
	initial_state = qstate(Complex{Float64}, L)
	hi = rand_herm_mat(L)
	hc = rand_herm_mat(L)

	m = ctrlham(hi, hc, rand(nparas), dt=0.1)

	circuit = build_rotational(L)

	loss(x, c) = real(dot(target_state, (x * (c * (x * (c * initial_state))))))

	return check_gradient(loss, m, circuit, dt=1.0e-8, atol=1.0e-4)
end

function ctrlham2_grad_dot_real(L::Int, nparas::Int)
	target_state = qrandn(Complex{Float64}, L)
	initial_state = qstate(Complex{Float64}, L)
	hi = rand_herm_mat(L)
	hc = rand_herm_mat(L)

	m = ctrlham(hi, [hc], nparas)

	loss(x) = real(dot(target_state, x * initial_state))

	return check_gradient(loss, m, dt=1.0e-8, atol=1.0e-4)
end

function ctrlham2_plus_T_grad_dot_real(L::Int, nparas::Int)
	target_state = qrandn(Complex{Float64}, L)
	initial_state = qstate(Complex{Float64}, L)
	hi = rand_herm_mat(L)
	hc = rand_herm_mat(L)

	m = ctrlham(hi, [hc], nparas)

	loss(x) = real(dot(target_state, x * initial_state)) + sum(get_times(x))

	return check_gradient(loss, m, dt=1.0e-8, atol=1.0e-4)
end

function ctrlham2_with_rx_grad_dot_real(L::Int, nparas::Int)
	target_state = qrandn(Complex{Float64}, L)
	initial_state = qstate(Complex{Float64}, L)
	hi = rand_herm_mat(L)
	hc = rand_herm_mat(L)

	m = ctrlham(hi, [hc], nparas)

	circuit = build_rotational(L)

	loss(x, c) = real(dot(target_state, (x * (c * (x * (c * initial_state))))))

	return check_gradient(loss, m, circuit, dt=1.0e-8, atol=1.0e-4)
end


function ctrlham3_grad_dot_real(L::Int, nparas::Int)
	target_state = qrandn(Complex{Float64}, L)
	initial_state = qstate(Complex{Float64}, L)
	hi = rand_herm_mat(L)
	hc = [rand_herm_mat(L), rand_herm_mat(L), rand_herm_mat(L)]

	m = ctrlham(hi, hc, nparas)

	loss(x) = real(dot(target_state, x * initial_state))

	return check_gradient(loss, m, dt=1.0e-8, atol=1.0e-4)
end

function ctrlham3_plus_T_grad_dot_real(L::Int, nparas::Int)
	target_state = qrandn(Complex{Float64}, L)
	initial_state = qstate(Complex{Float64}, L)
	hi = rand_herm_mat(L)
	hc = [rand_herm_mat(L)]

	m = ctrlham(hi, hc, nparas)

	loss(x) = real(dot(target_state, x * initial_state)) + sum(get_times(x))

	return check_gradient(loss, m, dt=1.0e-8, atol=1.0e-3)
end

function expham_with_rx_grad_dot_real(L::Int)
	target_state = qrandn(Complex{Float64}, L)
	initial_state = qstate(Complex{Float64}, L)

	m1 = expham(rand_herm_mat(L))
	m2 = expham(rand_herm_mat(L))
	circuit1 = build_rotational(L)
	circuit2 = build_rotational(L)

	loss(x1, x2, c1, c2) = real(dot(target_state, (x2 * (c2 * (x1 * (c1 * initial_state) )))))
	return check_gradient(loss, m1, m2, circuit1, circuit2, dt=1.0e-8, atol=1.0e-4)
end

@testset "gradient of quantum control hamiltonian with loss function real(dot(x, c*y))" begin
    for L in 2:4
       	for nparas in 5:2:11
			@test ctrlham_grad_dot_real(L, nparas)
			@test ctrlham2_grad_dot_real(L, nparas)
			@test ctrlham2_plus_T_grad_dot_real(L, nparas)

			@test ctrlham3_grad_dot_real(L, nparas)
			@test ctrlham3_plus_T_grad_dot_real(L, nparas)
		end 
    end
end

@testset "gradient of mixed quantum control and rotational layer with loss function real(dot(x, m*c*m*c*y))" begin
    for L in 2:4
       	for nparas in 10:2:20
			@test ctrlham_with_rx_grad_dot_real(L, nparas)
			@test ctrlham2_with_rx_grad_dot_real(L, nparas)
		end 
    end
end

@testset "gradient of mixed hamiltonian expm and rotational layer with loss function real(dot(x, m2*c2*m1*c1*y))" begin
    for L in 2:5
    	@test expham_with_rx_grad_dot_real(L)
    end
end

