
function to_digits(s::Vector{Int})
	r = 0.
	for i = 1:length(s)
		r = r + s[i]*(2.)^(-i)
	end
	return r
end

function phase_estimate_circuit(j::Vector{Int})
	L = length(j)
	circuit = QCircuit()
	phi = to_digits(j)
	U = [exp(2*pi*im*phi) 0; 0. 1.]
	for i = 1:L
		push!(circuit, (i, H))
	end

	tmp = U
	for i = L:-1:1
		push!(circuit, ((i, L+1), Gates.CONTROL(tmp)))
		tmp = tmp * tmp
	end
	push!(circuit, QFT(L)')
	return circuit
end


function simple_phase_estimation_1(L::Int, auto_reset::Bool=false)
	j = rand(0:1, L)
	state = StateVector(L+1)
	phi = to_digits(j)
	circuit = phase_estimate_circuit(j)
	apply!(circuit, state)
	res = Int[]
	for i = 1:(L+1)
		i, p = measure!(state, i, auto_reset=auto_reset)
		push!(res, i)
	end
	phi_out = to_digits(res)
	return (phi == phi_out) && (j[1:L] == res[1:L])
end

function simple_phase_estimation_1_dm(L::Int, auto_reset::Bool=false)
	j = rand(0:1, L)
	state = DensityMatrix(L+1)
	phi = to_digits(j)
	circuit = phase_estimate_circuit(j)
	apply!(circuit, state)
	res = Int[]
	for i = 1:(L+1)
		i, p = measure!(state, i, auto_reset=auto_reset)
		push!(res, i)
	end
	phi_out = to_digits(res)
	return (phi == phi_out) && (j[1:L] == res[1:L])
end

function simple_phase_estimation_2(L::Int)

	j = rand(0:1, L)
	state = StateVector(L+1)
	phi = to_digits(j)
	circuit = phase_estimate_circuit(j)
	apply!(circuit, state)
	res = Int[]
	for i = 1:(L+1)
		state, i, p = measure(state, 1)
		push!(res, i)
	end
	phi_out = to_digits(res)
	return length(storage(state))==1 && isapprox(state[1], 1, atol=1.0e-10) && (phi == phi_out) && (j[1:L] == res[1:L])
end

@testset "simple phase estimation" begin
    for L in 2:15
        @test simple_phase_estimation_1(L, false)
        @test simple_phase_estimation_1(L, true)
    end
    for L in 2:6
    	@test simple_phase_estimation_1_dm(L, false)
    	@test simple_phase_estimation_1_dm(L, true)
    end
    for L in 2:15
        @test simple_phase_estimation_2(L)
    end
end
