
"""
	variational_circuit_1d(L::Int, depth::Int, paras::Vector{<:Real})
Return a variational quantum circuit given L qubis and d depth
"""
function variational_circuit_1d(L::Int, depth::Int, paras::Vector{<:Real}=rand(variational_circuit_nparameters(L, depth)) .* 2π)
	(length(paras) == variational_circuit_nparameters(L, depth)) || throw("wrong number of parameters.")
	circuit = QCircuit()
	ncount = 1
	for i in 1:L
		push!(circuit, RzGate(i, paras[ncount], isparas=true))
		ncount += 1
		push!(circuit, RyGate(i, paras[ncount], isparas=true))
		ncount += 1
		push!(circuit, RzGate(i, paras[ncount], isparas=true))
		ncount += 1
	end
	for i in 1:depth
		for j in 1:(L-1)
		    push!(circuit, CNOTGate(j, j+1))
		end
		for j in 1:L
			push!(circuit, RzGate(j, paras[ncount], isparas=true))
			ncount += 1
			push!(circuit, RyGate(j, paras[ncount], isparas=true))
			ncount += 1
			push!(circuit, RzGate(j, paras[ncount], isparas=true))
			ncount += 1
		end
	end
	@assert ncount == length(paras)+1
	return circuit	
end
variational_circuit(args...) = variational_circuit_1d(args...)


function variational_circuit_2d(m::Int, n::Int, depth::Int, paras::Vector{<:Real}=rand(variational_circuit_nparameters(m*n, depth)) .* 2π)
	L = m*n
	(length(paras) == variational_circuit_nparameters(L, depth)) || throw("wrong number of parameters.")
	circuit = QCircuit()
	ncount = 1
	for i in 1:L
		push!(circuit, RzGate(i, paras[ncount], isparas=true))
		ncount += 1
		push!(circuit, RyGate(i, paras[ncount], isparas=true))
		ncount += 1
		push!(circuit, RzGate(i, paras[ncount], isparas=true))
		ncount += 1
	end	
	index = LinearIndices((m, n))
	for l in 1:depth
		for i in 1:m
		    for j in 1:(n-1)
		        push!(circuit, CNOTGate(index[i, j], index[i, j+1]))
		    end
		end
		for i in 1:(m-1)
		    for j in 1:n
		        push!(circuit, CNOTGate(index[i, j], index[i+1, j]))
		    end
		end
		for i in 1:L
			push!(circuit, RzGate(i, paras[ncount], isparas=true))
			ncount += 1
			push!(circuit, RyGate(i, paras[ncount], isparas=true))
			ncount += 1
			push!(circuit, RzGate(i, paras[ncount], isparas=true))
			ncount += 1
		end	
	end
	@assert ncount == length(paras)+1
	return circuit
end

variational_circuit_nparameters(L::Int, depth::Int) = (depth+1) * L * 3

