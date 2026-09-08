# VQCQuantumCircuitsExt.jl — QuantumCircuits IR 的接口/适配扩展
#
# 本扩展是整个 VQC 中**唯一**依赖 QuantumCircuits 的地方
# （`using QuantumCircuits` 时自动加载）：把 IR 指令
# （GateOp / ChannelOp / MeasOp / ReinitOp / BarrierOp / IfOp / BlockOp）
# 与 Pauli 代数桥接到核心层的通用原语
# （`apply` / `apply_kraus!` / `measure!` / …）上。
#
# 公开入口：`simulate` / `simulate!` / `apply!`（方法添加到 VQC 的存根
# 泛型函数上，由 VQC 导出）与 `ClassicalStore`；另扩展 QuantumCircuits
# 的 `nqubits` 与 `measure` 协议。

module VQCQuantumCircuitsExt

using LinearAlgebra
using QuantumCircuits
using QuantumCircuits: GateOp, ChannelOp, MeasOp, ReinitOp, BarrierOp, IfOp, BlockOp,
                       Operation, Circuit, Cond, Param, ClbitRef, CReg, ParamVector,
                       qubits, mat, kraus, parameters, unroll
import QuantumCircuits: nqubits, measure

using VQC
using VQC: StateVector, DensityMatrix, storage,
           apply, apply_kraus!, reset_qubit_zero!,
           expect_kernel, dm_expect_kernel,
           _lsb_key, _nqubits,
           measure!
import VQC: expectation, apply!, simulate, simulate!

include("quantumcircuits/ops.jl")
include("quantumcircuits/classical.jl")
include("quantumcircuits/simulate.jl")
include("quantumcircuits/hamiltonian.jl")

export ClassicalStore

end # module VQCQuantumCircuitsExt
