# VQC.jl

[QuantumCircuits](../QuantumCircuits) IR 的**态矢量 / 密度矩阵模拟后端**，附带
可插拔的自动微分扩展。

## 定位

在全栈量子软件分层中：

```
QuantumCircuits.jl   唯一中间表示（IR）：Gate / GateOp / ChannelOp / Circuit / QASM…
VQC.jl               模拟后端：把 IR 指令作用到 StateVector / DensityMatrix 上
    └── ext/VQCQuantumCircuitsExt  接口扩展（using QuantumCircuits 时自动加载）
    └── ext/VQCZygoteExt           自动微分（using Zygote + QuantumCircuits 时自动加载）
```

IR 只回答"是什么"，VQC 回答"怎么算"。VQC 的依赖全部走包扩展机制，
核心层零 IR / 零 AD 依赖：

```
src/                                                     核心层：态类型、
├── states/  kernels/  coreops.jl  ops/  hamiltonian.jl   局域矩阵/Kraus 原语、
│                                                         测量/后选择/偏迹
│                                                         （不引用任何 IR 类型）
ext/
├── VQCQuantumCircuitsExt.jl                              接口扩展（全包唯一
│   └── quantumcircuits/                                  依赖 IR 的地方）：
│                                                         apply!/simulate/经典控制/
│                                                         Pauli 期望值/nqubits·measure 协议扩展
└── VQCZygoteExt.jl                                       AD 扩展：一般参数门与
                                                          含噪线路的自动微分
```

## 特性

* **完整 IR 指令支持**：`GateOp`（任意酉门，含 `inv/pow/ctrl/negctrl` 修饰门）、
  `ChannelOp`（Kraus 噪声信道，噪声即指令）、`MeasOp` / `ReinitOp` /
  `BarrierOp`、经典条件分支 `IfOp`、结构块 `BlockOp`。
* **高性能核**：≤4 比特门走 `StaticArrays` 完全展开的专用核，更大门走通用核，
  大态自动多线程；密度矩阵、Pauli 期望值均有专用核。
* **符号参数**：`simulate(c, ψ; params=θ)` 按位绑定 `parameters(c)`，
  热循环免字典分配；同名参数即权重共享。
* **自动微分（包扩展）**：
  * 一般参数门的梯度——对 `ParamGate.matrix_fn` 做 Zygote 精确微分，
    不可微时自动回退中心差分（**不限于 Rx/Ry/Rz**）；
  * 噪声量子线路的梯度——信道超算符伴随/逆映射回传（`DensityMatrix` 全程）；
  * `expectation` / `post_select` / 初始态的梯度；权重共享自动累加。

## 快速上手

```julia
using VQC, QuantumCircuits
using QuantumCircuits.Hamiltonian: PauliTerm, PauliSum

c = Circuit(2)
push!(c, RX(:θ, 0))
push!(c, CX(0, 1))

ψ = simulate(c, zero_state(2); params=[0.3])
h = PauliSum([PauliTerm(1.0, 0 => :Z, 1 => :Z)])
expectation(h, ψ)

# 含噪线路
push!(c, Depolarizing(0, 1e-2))
ρ = simulate(c, DensityMatrix(zero_state(2)); params=[0.3])

# 自动微分
using Zygote
grad = Zygote.gradient(θ -> begin
    s = simulate(c, zero_state(2); params=[θ])
    real(expectation(h, s))
end, [0.3])
```

## 约定

* 比特索引 **1-based**、小端序（qubit 1 = 最低有效位），与 Julia /
  QuantumCircuits 一致；
* `apply!(state, op)` 就地作用单条指令；`simulate(circuit, state)` 非就地演化；
* 主包不依赖任何 AD 库——自动微分由 `ext/VQCZygoteExt.jl` 按
  Julia 包扩展机制在 `using Zygote` 时自动启用。
