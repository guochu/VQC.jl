

function check_qham_onebody_single(::Type{T}, L::Int, j::Int) where T
    state1 = rand_state(T, L)
    m = random_unitary(1)
    gate1 = QuantumGate(j, m)
    h = QubitsOperator(QubitsTerm(j=>m))
    state2 = h * state1
    apply!(gate1, state1)
    return maximum(abs.(storage(state1 - state2)) )
end

function check_qham_onebody(::Type{T}, L::Int) where {T<:Number}
    errs = []
    for c in 1:L
        push!(errs, check_qham_onebody_single(T, L, c))
    end
    # println("total number of tests $(length(errs)).")
    return all(errs .< 1.0e-6)
end

function check_qham_twobody_single(::Type{T}, L::Int, j::Int, k::Int) where T
    state1 = rand_state(T, L)
    m1 = random_unitary(1)
    m2 = random_unitary(1)
    gate1 = QuantumGate((j, k), kron(m1, m2))
    h = QubitsOperator(QubitsTerm(j=>m2, k=>m1))
    state2 = h * state1
    apply!(gate1, state1)
    return maximum(abs.(storage(state1 - state2)) )
end

function check_qham_twobody(::Type{T}, L::Int) where {T<:Number}
    errs = []
    for c in 1:2:L
        for t in 2:2:L
            if c != t
                push!(errs, check_qham_twobody_single(T, L, c, t))
            end
        end
    end
    # println("total number of tests $(length(errs)).")
    return all(errs .< 1.0e-6)
end

function check_qham_threebody_single(::Type{T}, L::Int, a::Int, b::Int, c::Int) where {T<:Number}
    state1 = rand_state(T, L)
    m1 = random_unitary(1)
    m2 = random_unitary(1)
    m3 = random_unitary(1)
    gate1 = QuantumGate((a, b, c), kron(m1, m2, m3))
    h = QubitsOperator(QubitsTerm(a=>m3, b=>m2, c=>m1))
    state2 = h * state1
    apply!(gate1, state1)
   	return maximum(abs.(storage(state1 - state2)) )
end

function check_qham_threebody(::Type{T}, L::Int) where {T<:Number}
    errs = []
    for a in 1:2:L
        for b in 2:3:L
            for c in 3:5:L
                if a != b && a != c && b != c
                    push!(errs, check_qham_threebody_single(T, L, a, b, c))
                end
            end
        end
    end
    # println("total number of tests $(length(errs)).")
    return all(errs .< 1.0e-6)
end

function check_term_matrix(L::Int)
    state = rand_state(ComplexF64, L)
    m = QubitsTerm(1=>"+", L-1=>"-", coeff=0.77)
    return maximum(abs.(storage(m * state - matrix(L, m) * state))) < 1.0e-6
end

function check_ham_matrix(L::Int)
    state = rand_state(ComplexF64, L)
    m = QubitsTerm(1=>"+", L-1=>"-", coeff=0.77) + QubitsTerm(1=>"-", L-1=>"+", coeff=0.77) + QubitsTerm(1=>"X", L=>"Y", coeff=0.36)
    return maximum(abs.(storage(m * state - matrix(L, m) * state))) < 1.0e-6
end

function check_term_expec(L::Int)
    state = rand_state(ComplexF64, L)
    m = QubitsTerm(1=>"+", L-1=>"-", coeff=0.77)
    return abs(expectation(m, state) - expectation(matrix(L, m), state)) < 1.0e-6
end

function check_ham_expec(L::Int)
    state = rand_state(ComplexF64, L)
    m = QubitsTerm(1=>"+", 2=>"-", coeff=0.77) + QubitsTerm(1=>"-", 2=>"+", coeff=0.77) + QubitsTerm(1=>"X", L-1=>"Y", coeff=0.36)
    return abs(expectation(m, state) - expectation(matrix(L, m), state)) < 1.0e-6
end

function check_ham_expec_long(L::Int, n::Int)
    state = rand_state(ComplexF64, L)
    m = QubitsTerm(Dict(i=>"X" for i in 1:n), coeff=0.77) + QubitsTerm(1=>"X", coeff=1.2) + QubitsTerm(1=>"X", 2=>"Z", coeff=0.3)
    return abs(expectation(m, state) - expectation(matrix(L, m), state)) < 1.0e-6
end

function check_expec_onebody_single(::Type{T}, L::Int, j::Int) where T
    state = rand_state(T, L)
    m = QubitsTerm(j=>random_unitary(1))

    return abs(expectation(m, state) - expectation(matrix(L, m), state))
end

function check_expec_onebody(::Type{T}, L::Int) where {T<:Number}
    errs = []
    for c in 1:L
        push!(errs, check_expec_onebody_single(T, L, c))
    end
    return all(errs .< 1.0e-6)
end

function check_expec_twobody_single(::Type{T}, L::Int, j::Int, k::Int) where T
    state = rand_state(T, L)
    m = QubitsTerm(j=>random_unitary(1), k=>random_unitary(1), coeff=0.73)

    return abs(expectation(m, state) - expectation(matrix(L, m), state))
end

function check_expec_twobody(::Type{T}, L::Int) where {T<:Number}
    errs = []
    for c in 1:2:L
        for t in 2:2:L
            if c != t
                push!(errs, check_expec_twobody_single(T, L, c, t))
            end
        end
    end
    # println("total number of tests $(length(errs)).")
    return all(errs .< 1.0e-6)
end

function check_expec_threebody_single(::Type{T}, L::Int, j::Int, k::Int, l::Int) where T
    state = rand_state(T, L)
    m = QubitsTerm(j=>random_unitary(1), k=>random_unitary(1), l=>random_unitary(1), coeff=0.9)

    return abs(expectation(m, state) - expectation(matrix(L, m), state))
end

function check_expec_threebody(::Type{T}, L::Int) where {T<:Number}
    errs = []
    for a in 1:L
        for b in 1:L
            for c in 1:L
                if a != b && a != c && b != c
                    push!(errs, check_expec_threebody_single(T, L, a, b, c))
                end
            end
        end
    end
    # println("total number of tests $(length(errs)).")
    return all(errs .< 1.0e-6)
end

function check_expec_onebody_single_2(::Type{T}, L::Int, j::Int) where T
    state1 = rand_state(T, L)
    state2 = rand_state(T, L)
    m = QubitsTerm(j=>random_unitary(1))

    return abs(expectation(state2, m, state1) - expectation(state2, matrix(L, m), state1))
end

function check_expec_onebody_2(::Type{T}, L::Int) where {T<:Number}
    errs = []
    for c in 1:L
        push!(errs, check_expec_onebody_single_2(T, L, c))
    end
    return all(errs .< 1.0e-6)
end

function check_expec_twobody_single_2(::Type{T}, L::Int, j::Int, k::Int) where T
    state1 = rand_state(T, L)
    state2 = rand_state(T, L)
    m = QubitsTerm(j=>random_unitary(1), k=>random_unitary(1), coeff=0.73)

    return abs(expectation(state2, m, state1) - expectation(state2, matrix(L, m), state1))
end

function check_expec_twobody_2(::Type{T}, L::Int) where {T<:Number}
    errs = []
    for c in 1:2:L
        for t in 2:2:L
            if c != t
                push!(errs, check_expec_twobody_single_2(T, L, c, t))
            end
        end
    end
    # println("total number of tests $(length(errs)).")
    return all(errs .< 1.0e-6)
end

function check_expec_threebody_single_2(::Type{T}, L::Int, j::Int, k::Int, l::Int) where T
    state1 = rand_state(T, L)
    state2 = rand_state(T, L)
    m = QubitsTerm(j=>random_unitary(1), k=>random_unitary(1), l=>random_unitary(1), coeff=0.9)

    return abs(expectation(state2, m, state1) - expectation(state2, matrix(L, m), state1))
end

function check_expec_threebody_2(::Type{T}, L::Int) where {T<:Number}
    errs = []
    for a in 1:L
        for b in 1:L
            for c in 1:L
                if a != b && a != c && b != c
                    push!(errs, check_expec_threebody_single_2(T, L, a, b, c))
                end
            end
        end
    end
    # println("total number of tests $(length(errs)).")
    return all(errs .< 1.0e-6)
end

@testset "check hamiltonion term expectation value" begin
    @test check_term_matrix(16)
    @test check_term_matrix(15)
    @test check_ham_matrix(6)
    @test check_ham_matrix(3)
    @test check_term_expec(16)
    @test check_term_expec(15)
    @test check_ham_expec(6)
    @test check_ham_expec(3)
    @test check_ham_expec_long(6, 5)
    @test check_ham_expec_long(6, 6)
    @test check_ham_expec_long(8, 8)
end

@testset "check generic hamiltonion term operations" begin
    @test check_qham_onebody(ComplexF32, 16)
    @test check_qham_onebody(ComplexF64, 15)
    for L in 5:10:15
        @test check_qham_twobody(ComplexF32, L)
        @test check_qham_twobody(ComplexF64, L)
    end
    @test check_qham_threebody(ComplexF32, 16)
    @test check_qham_threebody(ComplexF64, 15)

    @test (check_qham_onebody_single(ComplexF64, 2,2) < 1.0e-6)
    @test (check_qham_onebody_single(ComplexF64, 4,2) < 1.0e-6)
    @test (check_qham_twobody_single(ComplexF64, 2,2,1) < 1.0e-6)
    @test (check_qham_twobody_single(ComplexF64, 3,1,3) < 1.0e-6)
    @test (check_qham_threebody_single(ComplexF64, 3, 3,1,2) < 1.0e-6)
    @test (check_qham_threebody_single(ComplexF64, 4, 3,4,1) < 1.0e-6)
end


@testset "check generic hamiltonion expectation value" begin
    for L in 5:10:15
        @test check_expec_onebody(ComplexF64, L)
        @test check_expec_twobody(ComplexF64, L)
        @test check_expec_threebody(ComplexF64, L)

        @test check_expec_onebody_2(ComplexF64, L)
        @test check_expec_twobody_2(ComplexF64, L)
        @test check_expec_threebody_2(ComplexF64, L)
    end
end

