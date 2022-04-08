module Utilities

using VQC
using QuantumCircuits, QuantumCircuits.Gates

export heisenberg_1d, heisenberg_2d, ising_1d, ising_2d
export QFT
export variational_circuit, variational_circuit_1d, variational_circuit_2d
export order_finding

include("spin_hamiltonians.jl")
include("qft.jl")
include("variationalcircuit.jl")
include("shor/shor.jl")


end