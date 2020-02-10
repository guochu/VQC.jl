

abstract type AbstractQuantumOperation end
abstract type AbstractDifferentiableQuantumOperation <: AbstractQuantumOperation end

abstract type AbstractCircuit <: AbstractDifferentiableQuantumOperation end
abstract type AbstractGate <: AbstractDifferentiableQuantumOperation end
abstract type AbstractTransformedGate <: AbstractGate end

# hamiltonian exponential
abstract type AbstractHamiltonianExponential <: AbstractDifferentiableQuantumOperation end


