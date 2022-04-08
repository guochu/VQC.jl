

function check_ptrace(sites)
	a = rand_state(6)
	b = DensityMatrix(a)
	return partial_tr(a, sites) ≈ partial_tr(b, sites)
end

@testset "check partial trace of quantum state" begin
	@test check_ptrace([1,2])
	@test check_ptrace([2,5])
	@test check_ptrace([6,3])
end

