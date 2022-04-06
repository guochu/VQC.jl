
function test_parameter(L::Int, depth::Int)
	circuit = QCircuit()
	counts = 0
	vars = []
	for i in 1:L
	    push!(circuit, HGate(i))
	end
	for i in 1:L
		parameter = randn(Float64)
		# variable indicate a parameter
	    push!(circuit, RxGate(i, parameter, isparas=true))
	    counts += 1
	    push!(vars, parameter)
	    # this is not a parameter
	    push!(circuit, RxGate(i, parameter + 1, isparas=false))
	end
	for d in 1:depth
		for i in 1:(L-1)
	    	push!(circuit, CNOTGate(i, i+1))
		end
		for i in 1:L
			parameter = randn(Float64)
			# variable indicate a parameter
	    	push!(circuit, RxGate(i, parameter, isparas=true))
	    	counts += 1
	    	push!(vars, parameter)
	    	# this is not a parameter
	    	push!(circuit, RxGate(i, parameter + 1, isparas=false))
		end    
	end
	check1 = (nparameters(circuit) == counts)
	check2 = (active_parameters(circuit) == vars)
	new_vars = randn(Float64, size(vars)...)
	reset_parameters!(circuit, new_vars)
	check3 = (nparameters(circuit) == counts)
	check4 = (active_parameters(circuit) == new_vars)

	# println("check1 $check1, check2 $check2, check3 $check3, check4 $check4")
	return check1 && check2 && check3 && check4
end



@testset "set the parameters of quantum circuit" begin
	for L in 2:10
		for depth in 0:7
		    @test test_parameter(L, depth)
		end	    
	end
end



