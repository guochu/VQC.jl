"""
    VQC

QuantumCircuits IR 的**态矢量 / 密度矩阵模拟后端**。

分层：

* **核心层**（`src/states` / `src/kernels` / `src/coreops` / `src/ops` /
  `src/hamiltonian`）：态类型、局域矩阵 / Kraus 原语
  （`apply!` / `apply_kraus!`）、测量 / 后选择 / 偏迹、矩阵期望值；
* **QuantumCircuits 桥接层**（`src/quantumcircuits`）：把 IR 指令
  （`GateOp` / `ChannelOp` / `MeasOp` / `ReinitOp` / `BarrierOp` / `IfOp` /
  `BlockOp`）、Pauli 代数与自旋算符代数（`SpinOpTerm` / `SpinOpSum`，
  类型定义见 `QuantumCircuits.Hamiltonian`）桥接到核心原语；提供
  经典寄存器运行时（`ClassicalStore`）与 Interface 后端
  `StateVectorBackend`（`QuantumCircuits.simulate(c, backend; shots, seed)`）；
* **AD 扩展**（`ext/VQCZygoteExt.jl`，`using Zygote` 时自动加载）：
  一般参数门与含噪线路的自动微分。

约定（与 QuantumCircuits 一致）：

* 比特索引 **1-based**，小端序（qubit 1 = 最低有效位，与 Julia /
  QuantumCircuits 惯例一致）；
* 门矩阵约定：`GateOp.qubits` 列表中**第一个比特为矩阵最高位**。

核心接口：

* `apply!(state, op)`：把单条指令就地作用到态上；
* `simulate(circuit, state; params)`：演化整条线路（非就地，支持符号参数；
  挂在 `QuantumCircuits.Interface.simulate` 上）；
* `expectation(h, state)`：`PauliSum` / `PauliTerm` / `SpinOpTerm` /
  `SpinOpSum` / 一般矩阵的期望值；
* `probabilities` / `measure!` / `measure` / `sample` / `post_select` /
  `partial_tr`。
"""
module VQC

using LinearAlgebra
using Random
using StaticArrays
using QuantumCircuits
using QuantumCircuits: GateOp, ChannelOp, MeasOp, ReinitOp, BarrierOp, IfOp, BlockOp,
                       Operation, Circuit, Cond, Param, ParamVector, ClbitRef, CReg,
                       qubits, mat, kraus, parameters, unroll
import QuantumCircuits: nqubits, measure
using QuantumCircuits.Interface: Backend

# ── 态 ──
export StateVector, DensityMatrix, storage,
       zero_state, rand_state, rand_densitymatrix,
       onehot_encoding, qubit_encoding, amplitude_encoding, permute,
       reset!, reset_onehot!, reset_qubit!, amplitude, amplitudes

# ── 线性代数 / 信息量 ──
export fidelity, distance, distance2, schmidt_numbers, entropy, renyi_entropy

# ── 通用态原语 ──
export apply_kraus!, reset_qubit_zero!

# ── 非就地作用（自旋算符类型见 QuantumCircuits.Hamiltonian）──
export apply

# ── 测量 ──
export probabilities, marginal_probabilities, measure!,
       sample, post_select, post_select!

# ── 可观测量 / 约化 ──
export expectation, partial_tr

# ── QuantumCircuits 桥接层 ──
export ClassicalStore, StateVectorBackend

include("auxiliary/auxiliary.jl")
include("states/statevector.jl")
include("states/densitymatrix.jl")
include("kernels/kernels.jl")
include("coreops.jl")
include("ops/ops.jl")
include("hamiltonian.jl")
include("quantumcircuits/quantumcircuits.jl")

# ── 存根：IR 指令 / 自旋算符适配由 src/quantumcircuits 提供 ─────────────────

function apply! end
function apply end

export apply!

end # module
