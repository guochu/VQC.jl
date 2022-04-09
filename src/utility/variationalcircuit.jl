"""
	variational_circuit_1d(L::Int, depth::Int, paras::Vector{<:Real}; rotate_first::Bool)
Return a variational quantum circuit given L qubis and d depth
"""
function variational_circuit_1d(L::Int, depth::Int, paras::Vector{<:Real}=rand(variational_circuit_nparameters(ComplexF64, L, depth)) .* 2π; rotate_first::Bool=true)
	(length(paras) == variational_circuit_nparameters(ComplexF64, L, depth, rotate_first)) || throw("wrong number of parameters.")
	circuit = QCircuit()
	ncount = 1
	if rotate_first
		for i in 1:L
			push!(circuit, RzGate(i, paras[ncount], isparas=true))
			ncount += 1
			push!(circuit, RyGate(i, paras[ncount], isparas=true))
			ncount += 1
			push!(circuit, RzGate(i, paras[ncount], isparas=true))
			ncount += 1
		end		
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
variational_circuit(args...; kwargs...) = variational_circuit_1d(args...; kwargs...)

function real_variational_circuit_1d(L::Int, depth::Int, paras::Vector{<:Real}=rand(variational_circuit_nparameters(Float64, L, depth)) .* 2π; rotate_first::Bool=true)
	(length(paras) == variational_circuit_nparameters(Float64, L, depth, rotate_first)) || throw("wrong number of parameters.")
	circuit = QCircuit()
	ncount = 1
	if rotate_first
		for i in 1:L
			push!(circuit, RyGate(i, paras[ncount], isparas=true))
			ncount += 1
		end		
	end
	for i in 1:depth
		for j in 1:(L-1)
		    push!(circuit, CNOTGate(j, j+1))
		end
		for j in 1:L
			push!(circuit, RyGate(j, paras[ncount], isparas=true))
			ncount += 1
		end
	end
	@assert ncount == length(paras)+1
	return circuit	
end


function variational_circuit_2d(m::Int, n::Int, depth::Int, paras::Vector{<:Real}=rand(variational_circuit_nparameters(ComplexF64, m*n, depth)) .* 2π; rotate_first::Bool=true)
	L = m*n
	(length(paras) == variational_circuit_nparameters(ComplexF64, L, depth, rotate_first)) || throw("wrong number of parameters.")
	circuit = QCircuit()
	ncount = 1
	if rotate_first
		for i in 1:L
			push!(circuit, RzGate(i, paras[ncount], isparas=true))
			ncount += 1
			push!(circuit, RyGate(i, paras[ncount], isparas=true))
			ncount += 1
			push!(circuit, RzGate(i, paras[ncount], isparas=true))
			ncount += 1
		end	
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
variational_circuit_2d(shapes::Tuple{Int, Int}, args...; kwargs...) = variational_circuit_2d(shapes[1], shapes[2], args...; kwargs...)

function real_variational_circuit_2d(m::Int, n::Int, depth::Int, paras::Vector{<:Real}=rand(variational_circuit_nparameters(Float64, m*n, depth)) .* 2π; rotate_first::Bool=true)
	L = m*n
	(length(paras) == variational_circuit_nparameters(Float64, L, depth, rotate_first)) || throw("wrong number of parameters.")
	circuit = QCircuit()
	ncount = 1
	if rotate_first
		for i in 1:L
			push!(circuit, RyGate(i, paras[ncount], isparas=true))
			ncount += 1
		end			
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
			push!(circuit, RyGate(i, paras[ncount], isparas=true))
			ncount += 1
		end	
	end
	@assert ncount == length(paras)+1
	return circuit
end
real_variational_circuit_2d(shapes::Tuple{Int, Int}, args...; kwargs...) = real_variational_circuit_2d(shapes[1], shapes[2], args...; kwargs...)

function variational_circuit_nparameters(::Type{T}, L::Int, depth::Int, rotate_first::Bool=true) where {T <: Number}
	n = rotate_first ? depth+1 : depth
	return (T <: Real) ? n * L : n * L * 3
end 

