push!(LOAD_PATH, "../src")

include("util.jl")

using Test
using Zygote, LinearAlgebra

using QuantumCircuits, QuantumCircuits.Gates
using QuantumCircuits: n_qubits_mat_from_external
using VQC, VQC.Utilities
using VQC: apply_serial!

_check_unitary(m::AbstractMatrix) = isapprox(m' * m, LinearAlgebra.I) 

function from_external(key::NTuple{N, Int}, m::AbstractMatrix; check_unitary::Bool=true) where N
    (size(m, 1) == 2^N) || error("wrong input matrix size.")
    if check_unitary
        _check_unitary(m) || error("input matrix is not unitary.")
    end
    return QuantumGate(key, n_qubits_mat_from_external(m))
end  

function random_unitary(n::Int)
    L = 2^n
    m = randn(ComplexF64, L, L)
    return exp(im * (m + m'))
end


@testset "test quantum gate operations" begin
    include("check_gateop.jl")
    include("check_high_qubit_gate.jl")
end

@testset "test quantum state initialization" begin
    include("check_state_init.jl")
end


@testset "test qubit hamiltonian operations" begin
    include("check_qubits_ham.jl")
end

@testset "test circuit parameters" begin
    include("parameter.jl")
end

@testset "test measure and select" begin
    include("check_measure.jl")
end


@testset "test quantum algorithm" begin
    include("algs.jl")
end

@testset "test utility" begin
    include("check_utility.jl")
end


@testset "test quantum circuit gradient" begin
    include("circuitgrad.jl")
    include("crxgategrad.jl")
end


@testset "test expectation value gradient" begin
    include("check_ham_expec_diff.jl")
end

@testset "test quantum state gradient (may not variable via a quantum computer)" begin
    include("stategrad.jl")
end


@testset "test density matrix operations" begin
    include("stategrad.jl")
end

@testset "test noisy quantum circuit" begin
    include("check_noisy_grad.jl")
end

