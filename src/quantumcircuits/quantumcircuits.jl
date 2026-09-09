# quantumcircuits.jl — QuantumCircuits 桥接层（聚合入口）
#
# 把 IR 指令 / Pauli 代数 / 自旋算符代数桥接到核心原语，
# 并提供 Interface 后端 `StateVectorBackend`。

include("classical.jl")
include("ops.jl")
include("hamiltonian.jl")
include("simulate.jl")
include("spinop.jl")
include("backend.jl")
