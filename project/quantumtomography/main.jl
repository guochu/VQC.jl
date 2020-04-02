push!(LOAD_PATH, "../../src")

using VQC
using VQC: ZERO
using KrylovKit: eigsolve

using Zygote
using Zygote: @adjoint

using Optim
using JSON
using Flux
using Flux.Optimise

function parse_cmd_line_args(args::Vector{<:AbstractString}, s::AbstractString=":") 
	r = Dict{String, String}()
	for arg in args
		k, v = split(arg, s)
		r[k] = v
	end
	return r
end


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

function xxz_ground_state(L::Int, J::Real, Jzz::Real, h::Real)
	ham = Hamiltonian(L)
	for i in 1:L
		add!(ham, (i,), ("sz",), coeff=h) 
	end
	for i in 1:L-1
	    add!(ham, (i, i+1), ("sp", "sm"), coeff=2*J)
	    add!(ham, (i, i+1), ("sm", "sp"), coeff=2*J)
	    add!(ham, (i, i+1), ("sz", "sz"), coeff=Jzz)
	end
	eigval, eigvec, info = eigsolve(matrix(ham), 1, :SR; ishermitian=true, issymmetric=true)
	(info.converged >= 1) || error("eigsolve failed.")
	return eigvec[1]
end


function cswap_circuit(n::Int)
	circuit = QCircuit()
	add!(circuit, HGate(1))
	for i in 1:n
	    add!(circuit, FREDKINGate((1, 1+i, n+1+i)))
	end
	add!(circuit, HGate(1))
	return circuit
end


function swap_test(x::AbstractVector, y::AbstractVector)
	n = nqubits(x)
	circuit = cswap_circuit(n)
	full_state = qcat(ZERO, x, y)
	apply!(circuit, full_state)
	out_comes = Int[]
	prob = 0.
	nmeasure = 10000
	for i in 1:nmeasure   
		r, j, prob = measure(copy(full_state), 1)
		# println("j=$j, prob=$prob")
		push!(out_comes, j)
	end
	appro_prob = sum(out_comes) / length(out_comes)
	r = 1-2*appro_prob >= 0 ? sqrt(1-2*appro_prob) : 0.
	# println("exact fidelity is $(abs(cdot(x, y))), approximate fidelity is $r.")
	return r
end


function quantum_gradient_util(f::Function, x)
	r = []
	v0 = f(x)
	sca = 0.25 / v0 
	for i in 1:length(x)
	    # df = differentiate(x[i])
	    np = nparameters(x[i])
	    if np == 0
	        continue
	    elseif np == 1
	    	vs = collect_variables(x[i])
	    	x0 = vs[1] 
	    	# println("********************************************")
	    	set_parameters!([x0 + 0.5*pi], x[i])
	    	a = f(x)
	    	set_parameters!([x0 - 0.5*pi], x[i])
	    	b = f(x)
	    	# println("a=$a, b=$b.")
	    	push!(r, sca*(a * a - b * b))
	    	set_parameters!(vs, x[i])
	    else
	    	println("number of parameters in gate $(typeof(x[i])) is $np.")
	    	error("something wrong.")
	    end
	end	
	return v0, [r...]
end


function train_by_flux_approx(target_state, initial_state, c, x0, alpha, epochs)
	fidelity_approx(m) = swap_test(target_state, m * initial_state)
	fidelity_exact(m) = abs(vdot(target_state, m * initial_state))
	opt = ADAM(alpha)
	x0_tmp = copy(x0)
	circuit = copy(c)
	set_parameters!(x0_tmp, circuit)
	fvalues = [1 - fidelity_exact(circuit)]
	for i in 1:epochs
		ss, grad = quantum_gradient_util(fidelity_approx, circuit)
		Optimise.update!(opt, x0_tmp, -grad)
		set_parameters!(x0_tmp, circuit)
		# ss = loss_exact(circuit)
		ss = 1 - fidelity_exact(circuit)
		push!(fvalues, ss)
		if i % 5 == 0
			println("loss at the $i-th step is $ss.") 
		end
	end
	return parameters(circuit), fvalues
end

function train_by_flux_exact(target_state, initial_state, c, x0, alpha, epochs)
	loss_exact(m) = 1 - abs(vdot(target_state, m * initial_state))
	opt = ADAM(alpha)
	x0_tmp = copy(x0)
	circuit = copy(c)
	set_parameters!(x0_tmp, circuit)
	fvalues = [loss_exact(circuit)]
	for i in 1:epochs
		grad = collect_variables(gradient(loss_exact, circuit))
		Optimise.update!(opt, x0_tmp, grad)
		set_parameters!(x0_tmp, circuit)
		ss = loss_exact(circuit)
		push!(fvalues, ss)
		# println("loss at the $i-th step is $ss.")
	end
	return parameters(circuit), fvalues
end

function choose_initial_paras(f, circuit, L)
	x0 = randn(L)
	set_parameters!(x0, circuit)
	while f(circuit) > 0.5
	    x0 = randn(L)
	    set_parameters!(x0, circuit)
	end
	return x0
end

function train_by_flux_exact_n(target_state, initial_state, c, alpha, epochs, n, L)
	loss_exact(m) = 1 - abs(vdot(target_state, m * initial_state))
	x0 = choose_initial_paras(loss_exact, c, L)
	paras, fvalues = train_by_flux_exact(target_state, initial_state, c, x0, alpha, epochs)
	fidelity = fvalues[end]
	for i in 2:n
		x0_tmp = choose_initial_paras(loss_exact, c, L)
	    paras_tmp, fvalues_tmp = train_by_flux_exact(target_state, initial_state, c, x0_tmp, alpha, epochs)
	    if fvalues_tmp[end] < fidelity
	        paras = paras_tmp
	        fvalues = fvalues_tmp
	        x0 = x0_tmp
	    end
	end
	return paras, fvalues, x0
end

function learn_ground_state(paras)
	paras = parse_cmd_line_args(paras)
	L = parse(Int, get(paras, "L", "10"))
	# depth = parse(Int, get(paras, "depth", "4"))
	J::Float64 = parse(Float64, get(paras, "J", "1"))
	Jzz::Float64 = parse(Float64, get(paras, "Jzz", "0.5"))
	h::Float64 = parse(Float64, get(paras, "h", "0.1"))
	alpha::Float64 = parse(Float64, get(paras, "alpha", "0.05"))
	epochs::Int = parse(Int, get(paras, "epochs", "100"))
	n::Int = parse(Int, get(paras, "n", "5"))

	target_state = xxz_ground_state(L, J, Jzz, h)
	initial_state = qstate(Float64, [0 for _ in 1:L])

	
	loss_approx(m) = 1. - swap_test(target_state, m * initial_state)

	for depth in 1:5
		println("parameters used: L=$L, depth=$(depth), J=$J, Jzz=$Jzz, h=$h.")
		circuit = real_variational_circuit(L, depth)

		x0 = parameters(circuit)
		println("number of parameters: $(length(x0))")


		# x_opt, f_values = train_by_flux_approx(target_state, initial_state, circuit, x0, alpha, epochs)

		println("training exact.................")
		x_opt_exact, fvalues_exact, x0 = train_by_flux_exact_n(target_state, initial_state, circuit, alpha, epochs, n, length(x0))
		println("distance from exact $(fvalues_exact[end]).")

		println("training approximate.................")
		x_opt_approx, fvalues_approx = train_by_flux_approx(target_state, initial_state, circuit, x0, alpha, epochs)

		println("distance from approximate $(fvalues_approx[end]).")

		result = JSON.json(Dict("xexact"=>x_opt_exact, "xapprox"=>x_opt_approx, 
			"fvaluesexact"=>fvalues_exact, "fvaluesapprox"=>fvalues_approx))

		filename = "result/L" * string(L) * "depth" * string(depth) * "J" * string(J) * "Jzz" * string(Jzz) * "h" * string(h) * ".txt"
		println("save results to path $filename.")
		println()
		io = open(filename, "w")
		write(io, result)  
		# if fvalues_exact[end] <= 1.0e-2
		#      break
		# end 

	end
end



learn_ground_state(ARGS)





