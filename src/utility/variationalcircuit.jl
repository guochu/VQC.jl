export variational_circuit, variational_circuit_1d, variational_circuit_2d

function linear_index(shape::Tuple{Int, Int}, positions::Tuple{Int, Int}, rowmajor::Bool=true)
	i, j = positions
	(i<=shape[1] && j<=shape[2]) || error("index out of range.")
	if rowmajor 
		return (i-1)*shape[2] + j
	else
		return (j-1)*shape[1] + i
	end	
end


function variational_circuit_1d(L::Int, depth::Int, g::Function=rand)
	circuit = QCircuit()
	for i in 1:L
		add!(circuit, RzGate(i, Variable(g())))
		add!(circuit, RyGate(i, Variable(g())))
		add!(circuit, RzGate(i, Variable(g())))
	end
	for i in 1:depth
		for j in 1:(L-1)
		    add!(circuit, CNOTGate((j, j+1)))
		end
		for j in 1:L
			add!(circuit, RzGate(j, Variable(g())))
			add!(circuit, RyGate(j, Variable(g())))
			add!(circuit, RzGate(j, Variable(g())))
		end
	end
	return circuit	
end

variational_circuit(L::Int, depth::Int, g::Function=rand) = variational_circuit_1d(L, depth, g)


function variational_circuit_2d(m::Int, n::Int, depth::Int, g::Function=rand)
	L = m*n
	circuit = QCircuit()
	for i in 1:L
		add!(circuit, RzGate(i, Variable(g())))
		add!(circuit, RyGate(i, Variable(g())))
		add!(circuit, RzGate(i, Variable(g())))
	end	
	sp = (m, n)
	for l in 1:depth
		for i in 1:m
		    for j in 1:(n-1)
		        add!(circuit, CNOTGate((linear_index(sp, (i, j)), linear_index(sp, (i, j+1)))))
		    end
		end
		for i in 1:(m-1)
		    for j in 1:n
		        add!(circuit, CNOTGate((linear_index(sp, (i, j)), linear_index(sp, (i+1, j)))))
		    end
		end
		for i in 1:L
			add!(circuit, RzGate(i, Variable(g())))
			add!(circuit, RyGate(i, Variable(g())))
			add!(circuit, RzGate(i, Variable(g())))
		end	
	end
	return circuit
end
