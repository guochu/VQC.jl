



function check_fourbody_single(::Type{T}, L::Int, j::Int, k::Int, l::Int, m::Int) where T
    state1 = rand_state(T, L)
    m1 = random_unitary(1)
    m2 = random_unitary(1)
    m3 = random_unitary(1)
    m4 = random_unitary(1)
    gate1 = from_external((j, k, l, m), kron(m1, m2, m3, m4))
    h = QubitsTerm(j=>m1, k=>m2, l=>m3, m=>m4)
    state2 = matrix(L, h) * state1
    apply!(gate1, state1)
    return maximum(abs.(storage(state1 - state2)) )
end

function check_fourbody(::Type{T}, L::Int, i::Int, j::Int, k::Int, l::Int) where T
	errs = []
	push!(errs, check_fourbody_single(ComplexF64, L, i,j,k,l))
	push!(errs, check_fourbody_single(ComplexF64, L, i,j,l,k))
	push!(errs, check_fourbody_single(ComplexF64, L, j,i,k,l))
	push!(errs, check_fourbody_single(ComplexF64, L, j,i,l,k))
	push!(errs, check_fourbody_single(ComplexF64, L, l,k,j,i))
	push!(errs, check_fourbody_single(ComplexF64, L, k,i,l,j))
	return all(errs .< 1.0e-6)
end

function check_fivebody_single(::Type{T}, L::Int, j::Int, k::Int, l::Int, m::Int, n::Int) where T
    state1 = rand_state(T, L)
    m1 = random_unitary(1)
    m2 = random_unitary(1)
    m3 = random_unitary(1)
    m4 = random_unitary(1)
    m5 = random_unitary(1)
    gate1 = from_external((j, k, l, m, n), kron(m1, m2, m3, m4, m5))
    h = QubitsTerm(j=>m1, k=>m2, l=>m3, m=>m4, n=>m5)
    state2 = matrix(L, h) * state1
    apply!(gate1, state1)
    return maximum(abs.(storage(state1 - state2)) )
end

function check_fivebody(::Type{T}, L::Int, i::Int, j::Int, k::Int, l::Int, n::Int) where T
	errs = []
	push!(errs, check_fivebody_single(ComplexF64, L, i,j,k,l,n))
	push!(errs, check_fivebody_single(ComplexF64, L, i,j,l,n,k))
	push!(errs, check_fivebody_single(ComplexF64, L, j,i,n,k,l))
	push!(errs, check_fivebody_single(ComplexF64, L, j,n,i,l,k))
	push!(errs, check_fivebody_single(ComplexF64, L, l,k,j,n,i))
	push!(errs, check_fivebody_single(ComplexF64, L, n,k,i,l,j))
	return all(errs .< 1.0e-6)
end


function check_qham_fourbody_single(::Type{T}, L::Int, j::Int, k::Int, l::Int, m::Int) where T
    state1 = rand_state(T, L)
    m1 = random_unitary(1)
    m2 = random_unitary(1)
    m3 = random_unitary(1)
    m4 = random_unitary(1)
    gate1 = from_external((j, k, l, m), kron(m1, m2, m3, m4))
    h = QubitsTerm(j=>m1, k=>m2, l=>m3, m=>m4)
    state2 = h * state1
    apply!(gate1, state1)
    return maximum(abs.(storage(state1 - state2)) )
end

function check_qham_fourbody(::Type{T}, L::Int, i::Int, j::Int, k::Int, l::Int) where T
    errs = []
    push!(errs, check_qham_fourbody_single(ComplexF64, L, i,j,k,l))
    push!(errs, check_qham_fourbody_single(ComplexF64, L, i,j,l,k))
    push!(errs, check_qham_fourbody_single(ComplexF64, L, j,i,k,l))
    push!(errs, check_qham_fourbody_single(ComplexF64, L, j,i,l,k))
    push!(errs, check_qham_fourbody_single(ComplexF64, L, l,k,j,i))
    push!(errs, check_qham_fourbody_single(ComplexF64, L, k,i,l,j))
    return all(errs .< 1.0e-6)
end

function check_qham_fivebody_single(::Type{T}, L::Int, j::Int, k::Int, l::Int, m::Int, n::Int) where T
    state1 = rand_state(T, L)
    m1 = random_unitary(1)
    m2 = random_unitary(1)
    m3 = random_unitary(1)
    m4 = random_unitary(1)
    m5 = random_unitary(1)
    gate1 = from_external((j, k, l, m, n), kron(m1, m2, m3, m4, m5))
    h = QubitsTerm(j=>m1, k=>m2, l=>m3, m=>m4, n=>m5)
    state2 = h * state1
    apply!(gate1, state1)
    return maximum(abs.(storage(state1 - state2)) )
end


function check_qham_fivebody(::Type{T}, L::Int, i::Int, j::Int, k::Int, l::Int, n::Int) where T
    errs = []
    push!(errs, check_qham_fivebody_single(ComplexF64, L, i,j,k,l,n))
    push!(errs, check_qham_fivebody_single(ComplexF64, L, i,j,l,n,k))
    push!(errs, check_qham_fivebody_single(ComplexF64, L, j,i,n,k,l))
    push!(errs, check_qham_fivebody_single(ComplexF64, L, j,n,i,l,k))
    push!(errs, check_qham_fivebody_single(ComplexF64, L, l,k,j,n,i))
    push!(errs, check_qham_fivebody_single(ComplexF64, L, n,k,i,l,j))
    return all(errs .< 1.0e-6)
end

function check_expec_fourbody_single(::Type{T}, L::Int, j::Int, k::Int, l::Int, m::Int) where T
    state = rand_state(T, L)
    m = QubitsTerm(j=>random_unitary(1), k=>random_unitary(1), l=>random_unitary(1), 
        m=>random_unitary(1), coeff=0.9)

    return abs(expectation(m, state) - expectation(matrix(L, m), state))
end


function check_expec_fourbody(::Type{T}, L::Int, i::Int, j::Int, k::Int, l::Int) where T
    errs = []
    push!(errs, check_expec_fourbody_single(T, L, i,j,k,l))
    push!(errs, check_expec_fourbody_single(T, L, i,j,l,k))
    push!(errs, check_expec_fourbody_single(T, L, j,i,k,l))
    push!(errs, check_expec_fourbody_single(T, L, j,i,l,k))
    push!(errs, check_expec_fourbody_single(T, L, l,k,j,i))
    push!(errs, check_expec_fourbody_single(T, L, k,i,l,j))
    return all(errs .< 1.0e-6)
end

function check_expec_fourbody_single_2(::Type{T}, L::Int, j::Int, k::Int, l::Int, m::Int) where T
    state1 = rand_state(T, L)
    state2 = rand_state(T, L)
    m = QubitsTerm(j=>random_unitary(1), k=>random_unitary(1), l=>random_unitary(1), 
        m=>random_unitary(1), coeff=0.7)

    return abs(expectation(state2, m, state1) - expectation(state2, matrix(L, m), state1))
end

function check_expec_fourbody_2(::Type{T}, L::Int, i::Int, j::Int, k::Int, l::Int) where T
    errs = []
    push!(errs, check_expec_fourbody_single_2(T, L, i,j,k,l))
    push!(errs, check_expec_fourbody_single_2(T, L, i,j,l,k))
    push!(errs, check_expec_fourbody_single_2(T, L, j,i,k,l))
    push!(errs, check_expec_fourbody_single_2(T, L, j,i,l,k))
    push!(errs, check_expec_fourbody_single_2(T, L, l,k,j,i))
    push!(errs, check_expec_fourbody_single_2(T, L, k,i,l,j))
    return all(errs .< 1.0e-6)
end

function check_expec_fivebody_single(::Type{T}, L::Int, j::Int, k::Int, l::Int, m::Int, n::Int) where T
    state = rand_state(T, L)
    m = QubitsTerm(j=>random_unitary(1), k=>random_unitary(1), l=>random_unitary(1),
     m=>random_unitary(1), n=>random_unitary(1), coeff=0.77)

    return abs(expectation(m, state) - expectation(matrix(L, m), state))
end

function check_expec_fivebody_single_2(::Type{T}, L::Int, j::Int, k::Int, l::Int, m::Int, n::Int) where T
    state1 = rand_state(T, L)
    state2 = rand_state(T, L)
    m = QubitsTerm(j=>random_unitary(1), k=>random_unitary(1), l=>random_unitary(1),
     m=>random_unitary(1), n=>random_unitary(1), coeff=1.23)

    return abs(expectation(state2, m, state1) - expectation(state2, matrix(L, m), state1))
end

function check_expec_fivebody(::Type{T}, L::Int, i::Int, j::Int, k::Int, l::Int, n::Int) where T
    errs = []
    push!(errs, check_expec_fivebody_single(T, L, j,i,n,k,l))
    push!(errs, check_expec_fivebody_single(T, L, j,n,i,l,k))
    push!(errs, check_expec_fivebody_single(T, L, j,n,i,k,l))
    push!(errs, check_expec_fivebody_single(T, L, l,k,j,n,i))
    push!(errs, check_expec_fivebody_single(T, L, n,k,i,l,j))
    push!(errs, check_expec_fivebody_single(T, L, i,j,k,l,n))
    return all(errs .< 1.0e-6)
end

function check_expec_fivebody_2(::Type{T}, L::Int, i::Int, j::Int, k::Int, l::Int, n::Int) where T
    errs = []
    push!(errs, check_expec_fivebody_single_2(T, L, j,i,n,k,l))
    push!(errs, check_expec_fivebody_single_2(T, L, j,n,i,l,k))
    push!(errs, check_expec_fivebody_single_2(T, L, j,n,i,k,l))
    push!(errs, check_expec_fivebody_single_2(T, L, l,k,j,n,i))
    push!(errs, check_expec_fivebody_single_2(T, L, n,k,i,l,j))
    push!(errs, check_expec_fivebody_single_2(T, L, i,j,k,l,n))
    return all(errs .< 1.0e-6)
end

@testset "check high-qubit gate operations" begin
    @test check_fourbody(ComplexF32, 4,1,2,3,4)
    @test check_fourbody(ComplexF64, 10,10,7,1,4)
    @test check_fivebody(ComplexF32, 5,1,2,3,5,4)
    @test check_fivebody(ComplexF32, 10,9,7,10,1,4)
end

@testset "check high-qubit hamiltonian term operations" begin
    @test check_qham_fourbody(ComplexF32, 4,1,2,3,4)
    @test check_qham_fourbody(ComplexF64, 10,6,7,1,4)
    @test check_qham_fivebody(ComplexF32, 5,1,2,3,5,4)
    @test check_qham_fivebody(ComplexF32, 11,9,11,10,1,4)
end

@testset "check hamiltonion high-qubit term expectation value" begin
    @test check_expec_fourbody(ComplexF64, 4, 1,2,3,4)
    @test check_expec_fourbody(ComplexF64, 9, 7,8,1,5)
    @test check_expec_fivebody(ComplexF64, 5, 1,2,3,5,4)
    @test check_expec_fivebody(ComplexF64, 10, 7,10,2,5,8)
    @test check_expec_fourbody_2(ComplexF32, 4, 1,2,3,4)
    @test check_expec_fourbody_2(ComplexF64, 10, 7,10,1,5)
    @test check_expec_fivebody_2(ComplexF64, 5, 1,2,3,5,4)
    @test check_expec_fivebody_2(ComplexF32, 10, 7,10,2,5,8)
end
