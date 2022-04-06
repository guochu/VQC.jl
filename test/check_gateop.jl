



function check_onebody_single(::Type{T}, L::Int, c::Int) where {T<:Number}
    state1 = rand_state(T, L)
    state2 = copy(state1)
    gate = QuantumGate(c, random_unitary(1))
    apply_serial!(gate, state1)
    apply!(gate, state2)
    return maximum(abs.(storage(state1 - state2)) )
end

function check_onebody(::Type{T}, L::Int) where {T<:Number}
    errs = []
    for c in 1:L
        push!(errs, check_onebody_single(T, L, c))
    end
    # println("total number of tests $(length(errs)).")
    return all(errs .< 1.0e-6)
end

function check_twobody_single(::Type{T}, L::Int, c::Int, t::Int) where {T<:Number}
    state1 = rand_state(T, L)
    state2 = copy(state1)
    gate = QuantumGate((c, t), random_unitary(2))
    apply_serial!(gate, state1)
    apply!(gate, state2)
    return maximum(abs.(storage(state1 - state2)) )
end

function check_twobody(::Type{T}, L::Int) where {T<:Number}
    errs = []
    for c in 1:2:L
        for t in 2:2:L
            if c != t
                push!(errs, check_twobody_single(T, L, c, t))
            end
        end
    end
    # println("total number of tests $(length(errs)).")
    return all(errs .< 1.0e-6)
end

function check_threebody_single(::Type{T}, L::Int, a::Int, b::Int, c::Int) where {T<:Number}
    state1 = rand_state(T, L)
    state2 = copy(state1)
    gate = QuantumGate((a, b, c), random_unitary(3))
    apply_serial!(gate, state1)
    apply!(gate, state2)
    return maximum(abs.(storage(state1 - state2)) )
end

function check_threebody(::Type{T}, L::Int) where {T<:Number}
    errs = []
    for a in 1:2:L
        for b in 2:3:L
            for c in 3:5:L
                if a != b && a != c && b != c
                    push!(errs, check_threebody_single(T, L, a, b, c))
                end
            end
        end
    end
    # println("total number of tests $(length(errs)).")
    return all(errs .< 1.0e-6)
end

function check_single_gate(::Type{T}, L::Int, gt) where {T<:Number}
    state1 = rand_state(T, L)
    state2 = copy(state1)
    apply!(gt, state1)
    apply!(gate(ordered_positions(gt), ordered_mat(gt)), state2)
    return maximum(abs.(storage(state1 - state2)) )
end

function check_specialized_onebody_gates(::Type{T}, L::Int) where {T<:Number}
    errs = []
    for c in 1:L
        gt = XGate(c)
        push!(errs, check_single_gate(T, L, gt))
        push!(errs, check_single_gate(T, L, gt'))
        gt = YGate(c)
        push!(errs, check_single_gate(T, L, gt))
        push!(errs, check_single_gate(T, L, gt'))
        gt = ZGate(c)
        push!(errs, check_single_gate(T, L, gt))
        push!(errs, check_single_gate(T, L, gt'))
        gt = TGate(c)
        push!(errs, check_single_gate(T, L, gt))
        push!(errs, check_single_gate(T, L, gt'))
        gt = PHASEGate(c, randn())
        push!(errs, check_single_gate(T, L, gt))
        push!(errs, check_single_gate(T, L, gt'))
    end
    return all(errs .< 1.0e-6)
end

function check_specialized_twobody_gates(::Type{T}, L::Int) where {T<:Number}
    errs = []
    for c in 1:2:L
        for t in 2:2:L
            if c != t
                gt = CONTROLGate((c, t), random_unitary(1))
                push!(errs, check_single_gate(T, L, gt))
                push!(errs, check_single_gate(T, L, gt'))
                gt = CRxGate(c, t, randn())
                push!(errs, check_single_gate(T, L, gt))
                push!(errs, check_single_gate(T, L, gt'))
                gt = CRyGate(c, t, randn())
                push!(errs, check_single_gate(T, L, gt))
                push!(errs, check_single_gate(T, L, gt'))
                gt = CRzGate(c, t, randn())
                push!(errs, check_single_gate(T, L, gt))
                push!(errs, check_single_gate(T, L, gt'))

                gt = CZGate(c, t)
                push!(errs, check_single_gate(T, L, gt))
                push!(errs, check_single_gate(T, L, gt'))
                gt = CNOTGate(c, t)
                push!(errs, check_single_gate(T, L, gt))
                push!(errs, check_single_gate(T, L, gt'))

                gt = CPHASEGate(c, t, randn())
                push!(errs, check_single_gate(T, L, gt))
                push!(errs, check_single_gate(T, L, gt'))

                gt = SWAPGate(c, t)
                push!(errs, check_single_gate(T, L, gt))
                push!(errs, check_single_gate(T, L, gt'))
                gt = iSWAPGate(c, t)
                push!(errs, check_single_gate(T, L, gt))
                push!(errs, check_single_gate(T, L, gt'))

                gt = FSIMGate(c, t, randn(5))
                push!(errs, check_single_gate(T, L, gt))
                push!(errs, check_single_gate(T, L, gt'))
            end
        end
    end
    return all(errs .< 1.0e-6)
end

function check_specialized_threebody_gates(::Type{T}, L::Int) where {T<:Number}
    errs = []
    for a in 1:2:L
        for b in 2:3:L
            for c in 3:5:L
                if a != b && a != c && b != c
                    gt = CONTROLCONTROLGate(a, b, c, random_unitary(1))
                    push!(errs, check_single_gate(T, L, gt))
                    push!(errs, check_single_gate(T, L, gt'))
                    gt = TOFFOLIGate(a, b, c)
                    push!(errs, check_single_gate(T, L, gt))
                    push!(errs, check_single_gate(T, L, gt'))
                    gt = FREDKINGate(a, b, c)
                    push!(errs, check_single_gate(T, L, gt))
                    push!(errs, check_single_gate(T, L, gt'))

                    gt = CCPHASEGate(a, b, c, randn())
                    push!(errs, check_single_gate(T, L, gt))
                    push!(errs, check_single_gate(T, L, gt'))
                end
            end
        end
    end
    return all(errs .< 1.0e-6)
end

@testset "check generic gate operations" begin
    @test check_onebody(ComplexF32, 16)
    @test check_onebody(ComplexF64, 15)
    for L in 5:10:15
        @test check_twobody(ComplexF32, L)
        @test check_twobody(ComplexF64, L)
    end
    @test check_threebody(ComplexF32, 16)
    @test check_threebody(ComplexF64, 15)
end

@testset "check specialized gate operations" begin
    @test check_specialized_onebody_gates(ComplexF32, 16)
    @test check_specialized_twobody_gates(ComplexF32, 16)
    @test check_specialized_threebody_gates(ComplexF64, 15)
end
