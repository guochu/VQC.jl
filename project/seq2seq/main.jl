push!(LOAD_PATH, "../../src")

using VQC

using Zygote

const ncell = 3


function generate_single_circuit(L, d)
	circuit = QCircuit()
	for j in 1:L
	    push!(circuit, RzGate(j, Variable(randn())))
	    push!(circuit, RyGate(j, Variable(randn())))
	    push!(circuit, RzGate(j, Variable(randn())))
	end
	for i in 1:d
	    for j in 1:L-1
	        push!(circuit, CNOTGate((j, j+1)))
	    end
	    for j in 1:L
	    	push!(circuit, RzGate(j, Variable(randn())))
	    	push!(circuit, RyGate(j, Variable(randn())))
	    	push!(circuit, RzGate(j, Variable(randn())))	        
	    end
	end
	return circuit
end

function generate_circuits(d1, d2, d3, d4, d5, d6, d7)
	c1 = generate_single_circuit(2*ncell, d1)
	append!(c1, shift(generate_single_circuit(2*ncell, d2), ncell))
	append!(c1, shift(generate_single_circuit(2*ncell, d3), 2*ncell))
	append!(c1, shift(generate_single_circuit(2*ncell, d4), 3*ncell))
	c2 = shift(generate_single_circuit(2*ncell, d5), 3*ncell)
	c3 = shift(generate_single_circuit(2*ncell, d6), 3*ncell)
	c4 = shift(generate_single_circuit(2*ncell, d7), 3*ncell)
	return c1, c2, c3, c4
end


function main()
	N = 5
	
	nunits = 4
	Ls = nunits * ncell
	n_start = nunits * ncell
	fake_data = [(rand(0:1, Ls), rand(0:1, Ls)) for _ in 1:N]

	x_training = [qstate([a; [0 for _ in 1:ncell]]) for (a, b) in fake_data]
	y_training = [b for (a, b) in fake_data]	


	function loss_seq2seq(c1, c2, c3, c4)
		v = 0.
	
		for i in 1:length(x_training)
			p0 = 1.
		    tmp = c1 * x_training[i]
	    	for j in 1:ncell
	        	tmp, probability = post_select(tmp, n_start+j, y_training[i][j], keep=true)
	        	p0 = p0 * probability
		    end
		    tmp = c2 * tmp
		    for j in 1:ncell
		        tmp, probability = post_select(tmp, n_start+j, y_training[i][ncell+j], keep=true)
	    	    p0 = p0 * probability
		    end
		    tmp = c3 * tmp
	    	for j in 1:ncell
	        	tmp, probability = post_select(tmp, n_start+j, y_training[i][2*ncell+j], keep=true)
	        	p0 = p0 * probability
	    	end
	    	tmp = c4 * tmp
	    	for j in 1:ncell
	        	tmp, probability = post_select(tmp, n_start+j, y_training[i][3*ncell+j], keep=true)
	        	p0 = p0 * probability
	    	end
	    	v = v + p0
		end
		return v
	end

	# generate all the parametric quantum circuits
	c1, c2, c3, c4 = generate_circuits(2,3,2,2,5,5,2)

	# compute the loss
	@time v = loss_seq2seq(c1, c2, c3, c4)

	println("loss value is $v.")

	grad = gradient(loss_seq2seq, c1, c2, c3, c4)

	@time grad = gradient(loss_seq2seq, c1, c2, c3, c4)

	println("gradient is $(grad)")
end



main()







