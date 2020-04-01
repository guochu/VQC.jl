push!(LOAD_PATH, "../../src")

using VQC
using VQC: ZERO

using Zygote

function real_variational_circuit(L::Int, depth::Int)
	circuit = QCircuit()
	for i in 1:L
		add!(circuit, RyGate(i, Variable(randn(Float64))))
	end
	for i in 1:depth
		for j in 1:(L-1)
		    add!(circuit, CNOTGate((j, j+1)))
		end
		for j in 1:L
			add!(circuit, RyGate(j, Variable(randn(Float64))))
		end
	end
	return circuit	
end



function quantum_gradient_util(f::Function, x)
	r = []
	v0 = f(x)
	sca = 0.5 / v0
	for i in 1:length(x)
	    # df = differentiate(x[i])
	    np = nparameters(x[i])
	    if np == 0
	        continue
	    elseif np == 1
	    	vs = collect_variables(x[i])
	    	x0 = vs[1] 
	    	# println("********************************************")
	    	set_parameters!([x0 + 0.25*pi], x[i])
	    	a = f(x)
	    	set_parameters!([x0 - 0.25*pi], x[i])
	    	b = f(x)
	    	# println("a=$a, b=$b.")
	    	push!(r, sca*(a * a - b * b))
	    	set_parameters!(vs, x[i])
	    else
	    	error("something wrong.")
	    end
	end	
	return v0, [r...]
end



n = 3
target_state = qrandn(Float64, n)
initial_state = qstate(Float64, n)

circuit = real_variational_circuit(n, 1)

fidelity_exact(m) = abs(vdot(target_state, m * initial_state))

grad1 = collect_variables(gradient(fidelity_exact, circuit))
grad1 = [grad1...]

v, grad2 = quantum_gradient_util(fidelity_exact, circuit)

println("1--------------------------------------")
println(v)
println(fidelity_exact(circuit))

println("2--------------------------------------")
println(grad1 - grad2)





