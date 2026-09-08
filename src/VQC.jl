"""
    VQC

QuantumCircuits IR 的**态矢量 / 密度矩阵模拟后端**。

分层（对 QuantumCircuits 的依赖被完全隔离在 `ext/` 包扩展中，
核心层不引用任何 IR 类型）：

* **核心层**（`src/states` / `src/kernels` / `src/coreops` / `src/ops` /
  `src/hamiltonian`）：态类型、局域矩阵 / Kraus 原语
  （`apply!` / `apply_kraus!`）、测量 / 后选择 / 偏迹、矩阵期望值、
  自旋算符代数（`SpinOpTerm` / `SpinOpSum`，支持任意单比特算子的
  高效作用）；
* **接口扩展**（`ext/VQCQuantumCircuitsExt.jl`，`using QuantumCircuits`
  时自动加载）：把 IR 指令（`GateOp` / `ChannelOp` / `MeasOp` / `ReinitOp` /
  `BarrierOp` / `IfOp` / `BlockOp`）与 Pauli 代数桥接到核心原语，并提供
  `simulate` / 经典寄存器运行时 / `nqubits` 与 `measure` 协议扩展；
* **AD 扩展**（`ext/VQCZygoteExt.jl`，`using Zygote` + `using QuantumCircuits`
  时自动加载）：一般参数门与含噪线路的自动微分。

约定（与 QuantumCircuits 一致）：

* 比特索引 **1-based**，小端序（qubit 1 = 最低有效位，与 Julia /
  QuantumCircuits 惯例一致）；
* 门矩阵约定：`GateOp.qubits` 列表中**第一个比特为矩阵最高位**。

核心接口：

* `apply!(state, op)`：把单条指令就地作用到态上；
* `simulate(circuit, state; params)`：演化整条线路（非就地，支持符号参数）；
* `expectation(h, state)`：`PauliSum` / `PauliTerm` / 一般矩阵的期望值；
* `probabilities` / `measure!` / `measure` / `sample` / `post_select` /
  `partial_tr`。
"""
module VQC

using LinearAlgebra
using Random
using StaticArrays

# ── 核心层（不依赖 QuantumCircuits） ─────────────────────────────────────────

# ── 态 ──
export StateVector, DensityMatrix, storage,
       zero_state, rand_state, rand_densitymatrix,
       onehot_encoding, qubit_encoding, amplitude_encoding, permute,
       reset!, reset_onehot!, reset_qubit!, amplitude, amplitudes

# ── 线性代数 / 信息量 ──
export fidelity, distance, distance2, schmidt_numbers, entropy, renyi_entropy

# ── 通用态原语 ──
export apply_kraus!, reset_qubit_zero!

# ── 自旋算符代数（非就地 apply 见 hamiltonian.jl）──
export SpinOpTerm, SpinOpSum, apply

# ── 测量 ──
export probabilities, marginal_probabilities, measure!,
       sample, post_select, post_select!

# ── 可观测量 / 约化 ──
export expectation, partial_tr

include("auxiliary/auxiliary.jl")
include("states/statevector.jl")
include("states/densitymatrix.jl")
include("kernels/kernels.jl")
include("coreops.jl")
include("ops/ops.jl")
include("hamiltonian.jl")

# ── QuantumCircuits 接口存根 ─────────────────────────────────────────────────
#
# `simulate` / `simulate!` 在核心层只声明泛型函数存根，方法由扩展
# `VQCQuantumCircuitsExt`（`using QuantumCircuits` 时自动加载）提供；
# `apply!` 有两类方法：局域矩阵原语（coreops.jl）与 IR 指令适配（扩展）。

function simulate end
function simulate! end
function apply! end

export simulate, simulate!, apply!

end # module
