


function check_dm_expec_1() 
	psi = rand_state(ComplexF64, 7)
	rho = DensityMatrix(psi)

	h = QubitsTerm(1=>"X")

	a1 = expectation(h, psi)
	a2 = expectation(h, rho)

	return abs(a1 - a2) < 1.0e-6
end

function check_dm_expec_2() 
	psi = rand_state(ComplexF64, 7)
	rho = DensityMatrix(psi)

	h = QubitsTerm(1=>"X", 2=>"+")

	a1 = expectation(h, psi)
	a2 = expectation(h, rho)

	return abs(a1 - a2) < 1.0e-6
end


function check_dm_expec_3() 
	psi = rand_state(ComplexF64, 7)
	rho = DensityMatrix(psi)

	h = QubitsTerm(1=>"X", 2=>"+", 4=>"-")

	a1 = expectation(h, psi)
	a2 = expectation(h, rho)

	return abs(a1 - a2) < 1.0e-6
end

function check_dm_expec_4() 
	psi = rand_state(ComplexF64, 8)
	rho = DensityMatrix(psi)

	h = QubitsTerm(1=>"X", 2=>"+", 4=>"-", 8=>"Z")

	a1 = expectation(h, psi)
	a2 = expectation(h, rho)

	return abs(a1 - a2) < 1.0e-6
end


@testset "check density matrix expectation value" begin
    @test check_dm_expec_1()
    @test check_dm_expec_2()
    @test check_dm_expec_3()
    @test check_dm_expec_4()

end
