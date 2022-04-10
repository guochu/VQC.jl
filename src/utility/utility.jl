module Utilities

using VQC
using QuantumCircuits, QuantumCircuits.Gates

export heisenberg_1d, heisenberg_2d, ising_1d, ising_2d, ground_state, time_evolution
export QFT
export variational_circuit_1d, real_variational_circuit_1d
export order_finding

include("spin_hamiltonians.jl")
include("groundstate.jl")

include("qft.jl")
include("variationalcircuit.jl")
include("shor/shor.jl")


end