# classical.jl — IfOp 执行（经典寄存器运行时已上移至 QuantumCircuits）
#
# ClassicalStore / reg_value / set_clbit! / eval_cond 由
# QuantumCircuits 统一提供（2026-10 上移，三后端共享）；本文件只保留
# VQC 态上的 IfOp 分支执行与 QuantumCircuits 协议扩展。

using QuantumCircuits: eval_cond

"""
    apply!(state, op::IfOp, store::ClassicalStore) -> state

执行经典条件分支：按 `store` 求值 `op.cond`，执行 `op.then` 或
`op.otherwise` 子线路。需直接调用时请使用 `simulate`。
"""
function apply!(s::Union{StateVector,DensityMatrix}, op::IfOp, store::ClassicalStore,
                table::Union{Nothing,AbstractDict}=nothing)
    branch = eval_cond(op.cond, store) ? op.then : op.otherwise
    branch === nothing && return s
    for o in branch.ops
        s = apply!(s, o, table)
    end
    return s
end

function apply!(s::Union{StateVector,DensityMatrix}, op::IfOp, table::Union{Nothing,AbstractDict}=nothing)
    throw(ArgumentError("`apply!` on IfOp requires classical register values; use `simulate` instead"))
end

# ── QuantumCircuits 协议扩展：nqubits / measure ─────────────────────────────

"状态类型接入 QuantumCircuits 的 `nqubits` 协议。"
nqubits(x::StateVector) = _nqubits(x)
nqubits(x::DensityMatrix) = _nqubits(x)

"""
    measure(s::Union{StateVector,DensityMatrix}, q::Integer) -> (s', outcome, p)

QuantumCircuits `measure` 协议的态方法：非就地测量，返回坍缩后的
新态、测量结果与概率。
"""
function measure(s::Union{StateVector,DensityMatrix}, q::Integer)
    out = copy(s)
    marg = marginal_probabilities(out, Int(q))
    outcome = measure!(out, q)
    return out, outcome, marg[outcome + 1]
end
