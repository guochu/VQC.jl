using VQC: CRxGate, CNOTGate, Variable, nparameters


function crx_circuit(L::Int, depth::Int)
	n = 0
	circuit = QCircuit()
	for d in 1:depth
	   	for i in 1:L-1
	   		push!(circuit, CNOTGate((i, i+1)))
	   		push!(circuit, CRxGate((i, i+1), Variable(randn())))
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
	target_state = qrandn(Complex{Float64}, L)
	initial_state = qstate(Complex{Float64}, L)
	circuit, n =  crx_circuit(L, depth)

	loss(x) = real(dot(target_state, x * initial_state))

	return nparameters(circuit)==n && check_gradient(loss, circuit)
end

@testset "gradient of variable quantum circuit with parameterized controled rotational gate" begin
	for L in 2:5
		for depth in 1:5
		    @test crx_circuit_grad_dot_real(L, depth)
		end	    
	end
end