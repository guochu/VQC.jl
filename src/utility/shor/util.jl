include("binaryinteger.jl")


"""
	circuit to move i to j
"""
function move_site(i::Int, j::Int)
	circuit = QCircuit()
	(i==j) && return circuit
	if i < j
		for k = i:(j-1)
			# push!(circuit, ((k, k+1), SWAP))
			push!(circuit, SWAPGate(k, k+1))
		end
	else
		for k = i:-1:(j+1)
			# push!(circuit, ((k, k-1), SWAP))
			push!(circuit, SWAPGate(k, k-1))
		end
	end
	return circuit
end

# function compute_phase(a::BinaryInteger, j::Int)
# 	s = 1
# 	n = length(a)
# 	for k in 1:(j+1)
# 		aj = a[n-(j+1-k)]
# 		s = s*exp(aj*pi*im/2^(k-1))
#     end
# 	return s
# end
# rotationA(a::BinaryInteger, j::Int) = [1 0; 0 compute_phase(a, j)]
function compute_phase(a::BinaryInteger, j::Int)
	s = 0.
	n = length(a)
	for k in 1:(j+1)
		aj = a[n-(j+1-k)]
		s += aj / 2^(k-1)
    end
	return s * pi
end

function control_swap_block(c::Int, n::Int, i::Int, inci::Int, j::Int, incj::Int)
	swap_i_j = QCircuit()
	# c_cnot = _row_kron(UP, eye(4)) + _row_kron(DOWN, CNOT)
	(i == j) && error("position not allowed.")
	for k in 0:(n-1)
        # push!(swap_i_j, gate((j+k*incj, i+k*inci), CNOT))
		push!(swap_i_j, CNOTGate(j+k*incj, i+k*inci))
        # push!(swap_i_j, gate((c, i+k*inci, j+k*incj), c_cnot))
		push!(swap_i_j, TOFFOLIGate(c, i+k*inci, j+k*incj))
        # push!(swap_i_j, gate((j+k*incj, i+k*inci), CNOT))
		push!(swap_i_j, CNOTGate(j+k*incj, i+k*inci))
    end
	return swap_i_j
end


_QFT(L::Int) = shift(QFT(L), -1)
