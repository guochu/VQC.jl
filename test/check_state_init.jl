
function check_statevector_init_1()
    a0 = storage(StateVector(1)) == Gates.ZERO
    a1 = storage(onehot_encoding([1])) == Gates.ONE
    a2 = storage(onehot_encoding([0])) == Gates.ZERO
    a3 = storage(onehot_encoding(2)) == kron(Gates.ZERO, Gates.ZERO)
    a4 = storage(onehot_encoding([1, 0])) == kron(Gates.ZERO, Gates.ONE)
    a5 = storage(onehot_encoding([0, 1])) == kron(Gates.ONE, Gates.ZERO)
    a6 = storage(onehot_encoding([1, 1])) == kron(Gates.ONE, Gates.ONE)
    a7 = storage(onehot_encoding([1,0,0])) == kron(Gates.ZERO, Gates.ZERO, Gates.ONE)
    a8 = storage(reset_onehot!(onehot_encoding([1, 1]), [0, 1])) == kron(Gates.ONE, Gates.ZERO)
    a9 = storage(reset_onehot!(onehot_encoding([1,0,1]), [0,0,1])) == kron(Gates.ONE, Gates.ZERO, Gates.ZERO)
    return a0 && a1 && a2 && a3 && a4 && a5 && a6 && a7 && a8 && a9
end 


function check_statevector_init_2()
    v = [sqrt(2)/2, sqrt(2)/2]
    a0 = isapprox(onehot_encoding([0]), qubit_encoding([0.]) )
    a1 = isapprox(onehot_encoding([1]), qubit_encoding([1.]))
    a2 = isapprox(onehot_encoding([1,0]), qubit_encoding([1.,0.]))
    a3 = isapprox(onehot_encoding([1,1]), qubit_encoding([1.,1.]))
    a4 = isapprox(storage(qubit_encoding([0.5])), v)
    a5 = isapprox(storage(qubit_encoding([0.5, 1])), kron(Gates.ONE, v))
    a6 = isapprox(storage(reset_qubit!(onehot_encoding([0, 1]), [0, 0.5])), kron(v, Gates.ZERO))
    a7 = isapprox(storage(reset_qubit!(onehot_encoding([1,1,1]), [0, 0.5, 0.5])), kron(v, v, Gates.ZERO))
    a8 = isapprox(storage(reset_qubit!(onehot_encoding([1,1,1,0,1]), [0, 0.5, 0.5, 1, 0])), kron(Gates.ZERO, Gates.ONE, v, v, Gates.ZERO))
    return a0 && a1 && a2 && a3 && a4 && a5 && a6 && a7 && a8
end


function check_statevector_init_3(L::Int)
	v = randn(L)
	a0 = isapprox(StateVector(qubit_encoding(v)), qubit_encoding(v), atol=1.0e-8)
	v = rand(0:1, L)
	a2 = isapprox(StateVector(onehot_encoding(v)), onehot_encoding(v), atol=1.0e-8)
	return a0 && a2
end


@testset "check qubit encoding initialization of quantum state" begin
	@test check_statevector_init_1()
	@test check_statevector_init_2()
	for L in [3,4]
	    @test check_statevector_init_3(L)
	end
end
