# hamiltonian.jl — 核心层：一般矩阵期望值
#
# Pauli 代数（PauliTerm / PauliSum）的期望值见
# `src/quantumcircuits/hamiltonian.jl`（QuantumCircuits 接口层）。

"""
    expectation(m::AbstractMatrix, s::Union{StateVector,DensityMatrix}) -> Complex

一般（可显式构造的）算符期望值：`⟨ψ|m|ψ⟩` 或 `tr(ρ m)`。
大矩阵直接走 BLAS；小局域算符建议用局域核（见 Pauli 接口层方法）。
"""
function expectation(m::AbstractMatrix, s::StateVector)
    size(m, 1) == size(m, 2) == length(s) || throw(DimensionMismatch("matrix size mismatch"))
    return dot(storage(s), m, storage(s))
end

function expectation(m::AbstractMatrix, s::DensityMatrix)
    d = 1 << _nqubits(s)
    size(m, 1) == size(m, 2) == d || throw(DimensionMismatch("matrix size mismatch"))
    return sum(vec(storage(s)) .* vec(transpose(m)))   # tr(ρ m) = Σ ρ_ij m_ji
end

"""
    expectation(state_c::StateVector, m::AbstractMatrix, state::StateVector) -> Complex

一般矩阵元 `⟨state_c|m|state⟩`（供自动微分与振幅放大使用）。
"""
function expectation(state_c::StateVector, m::AbstractMatrix, state::StateVector)
    length(state_c) == length(state) == size(m, 1) == size(m, 2) ||
        throw(DimensionMismatch("dimension mismatch"))
    return dot(storage(state_c), m, storage(state))
end
