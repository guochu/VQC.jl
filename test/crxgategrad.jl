

function crx_circuit(L::Int, depth::Int)
	n = 0
	circuit = QCircuit()
	for d in 1:depth
	   	for i in 1:L-1
	   		push!(circuit, CNOTGate((i, i+1)))
	   		push!(circuit, CRxGate((i, i+1), randn(), isparas=true))
	   		push!(circuit, CNOTGate((i, i+1)))
	   		n += 1
		end 
	end
	return circuit, n
end

"""
	circuit gradient with dot loss function
"""
function crx_circuit_grad_dot_real(L::Int, depth::Int)
	target_state = rand_state(ComplexF64, L)
	initial_state = StateVector(ComplexF64, L)
	circuit, n =  crx_circuit(L, depth)

	loss(x) = real(dot(target_state, x * initial_state))
	loss_fd(θs) = begin
		reset_parameters!(circuit, θs)
		return loss(circuit)
	end

	grad1 = gradient(loss, circuit)[1]
	grad2 = fdm_gradient(loss_fd, active_parameters(circuit))

	return nparameters(circuit)==n && maximum(abs.(grad1 - grad2)) < 1.0e-6
end


function gfsim1_circuit(L::Int, depth::Int)
	circuit = QCircuit()
	for i in 1:L
	    push!(circuit, HGate(i))
	end
	for d in 1:depth
	   	for i in 1:L-1
	   		push!(circuit, CNOTGate((i, i+1)))
	   		push!(circuit, FSIMGate((i, i+1), randn(5), isparas=[true, false, false, false, false]))
	   		push!(circuit, CNOTGate((i, i+1)))
		end 
	end
	return circuit
end

function gfsim2_circuit(L::Int, depth::Int)
	circuit = QCircuit()
	for i in 1:L
	    push!(circuit, HGate(i))
	end
	for d in 1:depth
	   	for i in 1:L-1
	   		push!(circuit, CNOTGate((i, i+1)))
	   		push!(circuit, FSIMGate((i, i+1), randn(5), isparas=[true, true, false, false, false]))
	   		push!(circuit, CNOTGate((i, i+1)))
		end 
	end
	return circuit
end

function gfsim5_circuit(L::Int, depth::Int)
	circuit = QCircuit()
	for i in 1:L
	    push!(circuit, HGate(i))
	end
	for d in 1:depth
	   	for i in 1:L-1
	   		push!(circuit, CNOTGate((i, i+1)))
	   		push!(circuit, FSIMGate((i, i+1), randn(5), isparas=[true, true, true, true, true]))
	   		push!(circuit, CNOTGate((i, i+1)))
		end 
	end
	return circuit
end

function bf_circuit_gfims1_grad_dot_abs(L::Int, depth::Int)
	target_state = rand_state(ComplexF64, L)
	initial_state = StateVector(ComplexF64, L)
	circuit =  gfsim1_circuit(L, depth)

	loss(x) = abs(dot(target_state, x * initial_state))
	loss_fd(θs) = begin
		reset_parameters!(circuit, θs)
		return loss(circuit)
	end

	grad1 = gradient(loss, circuit)[1]
	grad2 = fdm_gradient(loss_fd, active_parameters(circuit))
	return maximum(abs.(grad1 - grad2)) < 1.0e-6
end

function bf_circuit_gfims2_grad_dot_abs(L::Int, depth::Int)
	target_state = rand_state(ComplexF64, L)
	initial_state = StateVector(ComplexF64, L)
	circuit =  gfsim2_circuit(L, depth)

	loss(x) = abs(dot(target_state, x * initial_state))
	loss_fd(θs) = begin
		reset_parameters!(circuit, θs)
		return loss(circuit)
	end

	grad1 = gradient(loss, circuit)[1]
	grad2 = fdm_gradient(loss_fd, active_parameters(circuit))
	return maximum(abs.(grad1 - grad2)) < 1.0e-6
end

function bf_circuit_gfims5_grad_dot_abs(L::Int, depth::Int)
	target_state = rand_state(ComplexF64, L)
	initial_state = StateVector(ComplexF64, L)
	circuit =  gfsim5_circuit(L, depth)

	loss(x) = abs(dot(target_state, x * initial_state))
	loss_fd(θs) = begin
		reset_parameters!(circuit, θs)
		return loss(circuit)
	end

	grad1 = gradient(loss, circuit)[1]
	grad2 = fdm_gradient(loss_fd, active_parameters(circuit))
	return maximum(abs.(grad1 - grad2)) < 1.0e-6
end


@testset "gradient of variable quantum circuit with parameterized controled rotational gate" begin
	for L in 2:5
		for depth in 1:5
		    @test crx_circuit_grad_dot_real(L, depth)
		end	    
	end
end

@testset "gradient of brute force quantum circuit with FSIM gate" begin
	for L in 2:5
		for depth in 1:3
		    @test bf_circuit_gfims1_grad_dot_abs(L, depth)
		    @test bf_circuit_gfims2_grad_dot_abs(L, depth)
		    @test bf_circuit_gfims5_grad_dot_abs(L, depth)
		end
	end
end

