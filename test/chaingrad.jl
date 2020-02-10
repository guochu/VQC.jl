push!(LOAD_PATH, "../src")


using VQC: ctrlham, qrandn, qstate, check_gradient, RxGate, expham, QCircuit, Variable, Chain

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


function chain_grad_dot_real_1(L::Int, nparas::Int)
	target_state = qrandn(Complex{Float64}, L)
	initial_state = qstate(Complex{Float64}, L)
	hi = rand_herm_mat(L)
	hc = rand_herm_mat(L)

	m = ctrlham(hi, hc, rand(nparas), dt=0.1)

	circuit = build_rotational(L)



	loss(x, c) = real(dot(target_state, Chain(c, x, c, x) * initial_state))

	grad = gradient(loss, m, circuit)
	return check_gradient(loss, m, circuit, dt=1.0e-8, atol=1.0e-4)
end

function chain_grad_dot_real_2(L::Int)
	target_state = qrandn(Complex{Float64}, L)
	initial_state = qstate(Complex{Float64}, L)

	m1 = expham(rand_herm_mat(L))
	m2 = expham(rand_herm_mat(L))
	circuit1 = build_rotational(L)
	circuit2 = build_rotational(L)

	chain = Chain(circuit1, m1, circuit2, m2)


	loss(m) = real(dot(target_state, m * initial_state ))


	return check_gradient(loss, chain, dt=1.0e-8, atol=1.0e-4)
end


@testset "gradient of chain with loss function real(dot(x, Chain(m,c,m,c)*y))" begin
    for L in 2:4
       	for nparas in 10:10:100
			@test chain_grad_dot_real_1(L, nparas)
		end 
    end
end

@testset "gradient of chain with loss function real(dot(x, Chain(m2,c2,m1,c1)*y))" begin
    for L in 2:5
    	@test chain_grad_dot_real_2(L)
    end
end

