# hamiltonian.jl — Pauli 代数与 GateOp 可观测量的期望值（接口层）

using QuantumCircuits.Hamiltonian: PauliTerm, PauliSum

const _PAULI1 = Dict{Symbol,Matrix{ComplexF64}}(
    :I => [1 0; 0 1],
    :X => [0 1; 1 0],
    :Y => [0 -im; im 0],
    :Z => [1 0; 0 -1],
)

"""
Pauli 项的局域矩阵：局域索引按 `ops` 中 qubit 顺序小端展开
（`ops[1]` 为最低位），与 `kron(P_{ops[N]}, …, P_{ops[1]})` 一致。
"""
function _pauli_local_matrix(ops::Vector{Pair{Int,Symbol}})
    isempty(ops) && throw(ArgumentError("empty Pauli ops"))
    m = _PAULI1[ops[end][2]]
    for i in length(ops)-1:-1:1
        m = kron(m, _PAULI1[ops[i][2]])
    end
    return m
end

"""
    expectation(t::PauliTerm, s::Union{StateVector,DensityMatrix}) -> Complex

Pauli 项的期望值：纯态 `⟨ψ|t|ψ⟩`；混合态 `tr(t ρ)`。
恒等项返回 `t.coeff * ⟨ψ|ψ⟩`（或 `t.coeff * tr(ρ)`）。
"""
function expectation(t::PauliTerm, s::StateVector)
    isempty(t.ops) && return t.coeff * dot(s, s)
    qs = Int[q for (q, _) in t.ops]                    # 1-based，用户给定顺序
    all(q -> 1 <= q <= _nqubits(s), qs) ||
        throw(ArgumentError("Pauli term qubit index out of range for state with $(_nqubits(s)) qubit(s)"))
    m = _pauli_local_matrix(t.ops)
    val = expect_kernel(storage(s), ntuple(i -> qs[i] - 1, Val(length(qs))), m)
    return t.coeff * val
end

function expectation(t::PauliTerm, s::DensityMatrix)
    isempty(t.ops) && return t.coeff * tr(s)
    qs = Int[q for (q, _) in t.ops]
    all(q -> 1 <= q <= _nqubits(s), qs) ||
        throw(ArgumentError("Pauli term qubit index out of range for state with $(_nqubits(s)) qubit(s)"))
    m = _pauli_local_matrix(t.ops)
    val = dm_expect_kernel(s.data, 1 << _nqubits(s), ntuple(i -> qs[i] - 1, Val(length(qs))), m)
    return t.coeff * val
end

"""
    expectation(h::PauliSum, s::Union{StateVector,DensityMatrix}) -> Complex

Pauli 和的期望值（逐项求和）。
"""
function expectation(h::PauliSum, s::Union{StateVector,DensityMatrix})
    acc = zero(ComplexF64)
    for t in h.terms
        acc += expectation(t, s)
    end
    return acc
end

"""
    expectation(op::GateOp, s::Union{StateVector,DensityMatrix}) -> Complex

把酉门 `op` 当作可观测量：`⟨ψ|U|ψ⟩` 或 `tr(ρ U)`。
"""
function expectation(op::GateOp, s::StateVector)
    return expect_kernel(storage(s), _lsb_key(Tuple(qubits(op))), mat(op))
end

function expectation(op::GateOp, s::DensityMatrix)
    return dm_expect_kernel(s.data, 1 << _nqubits(s), _lsb_key(Tuple(qubits(op))), mat(op))
end
