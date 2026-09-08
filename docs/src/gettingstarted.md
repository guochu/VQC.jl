# 快速开始

本节给出与 QuantumCircuits 协同做量子线路模拟的基本流程：
构造态 → 构造线路 → 演化 → 测量。VQC 自身的核心层接口
（不依赖 IR、可直接操作量子态的原语）系统列表见
[核心层 API](@ref)。

## 比特与索引约定（internal index convention）

VQC 与 QuantumCircuits 共用同一套索引约定，全部接口一致：

* **比特编号**：1-based、**小端序**——qubit 1 是最低有效位（LSB），
  与 `2^i` 权重的直觉对应：qubit `i` 的权重为 `2^(i-1)`。
* **振幅下标**：计算基矢 `|qₙ…q₂q₁⟩` 的振幅存储在
  `storage(ψ)[1 + q₁ + 2·q₂ + … + 2^(n-1)·qₙ]`（1-based 数组下标）。
* **门矩阵的位序**：`positions` / `qubits` 列表中**第一个比特是矩阵的
  最高位**（MSB-first）：`apply!(ψ, M, (2, 1))` 中 `M` 的行列索引
  按 `idx = 2·q₂ + q₁` 解释，即 `M` 的最高位作用在 qubit 2、最低位
  作用在 qubit 1。
* **密度矩阵**：平坦存储为列主序（行索引是低位），等价于
  `vec(ρ)`；`storage(ρ)` 直接返回 `2ⁿ×2ⁿ` 矩阵视图。
* **内部 kernel**：位运算统一用 0-based LSB-first 的位号
  （内部经 `_lsb_key` 转换），普通用户无需关心。

小端序验证：对 qubit 2 作用 `X` 门，唯一的非零振幅在 1-based 下标
`1 + 0·2⁰ + 1·2¹ = 3`（`q₁ = 0, q₂ = 1`）：

```@example
using QuantumCircuits, VQC

c = Circuit(2)
push!(c, X(2))
ψ = simulate(c, zero_state(2))
probabilities(ψ)          # = [0, 0, 1, 0]：非零振幅在下标 3
```

门矩阵位序验证：`M = [0 1 0 0; 1 0 0 0; 0 0 0 1; 0 0 1 0]` 作用在
`positions = (2, 1)` 上等价于交换 qubit 1 与 qubit 2（`M` 的
`|10⟩↔|01⟩` 块沿 (q₂,q₁) 索引）——这正是 `SWAP` 门的行为：

```@example
using QuantumCircuits, VQC

M = [0 1 0 0; 1 0 0 0; 0 0 0 1; 0 0 1 0]   # 行列索引 = 2·q₂ + q₁
c = Circuit(2)
push!(c, SWAP(1, 2))
ψ1 = simulate(c, onehot_encoding([1, 0]))   # SWAP 交换两个比特
ψ2 = copy(onehot_encoding([1, 0]))
apply!(ψ2, M, (2, 1))                       # 等价的局域矩阵作用
ψ1 ≈ ψ2
```

## 量子态

态矢量为 `StateVector`（纯态），密度矩阵为 `DensityMatrix`（混合态），
由 `zero_state` / `rand_state` / `onehot_encoding` 等构造
（完整列表见[核心层 API](@ref)）。

```@example
using VQC

ψ0 = zero_state(3)               # |000⟩
ψp = qubit_encoding([π/2, 0.0, 0.0])   # 单比特旋转角直积态
ρ = DensityMatrix(zero_state(2)) # 由纯态构造密度矩阵
amplitude(ψp, [1, 0, 0])         # ⟨100|ψp⟩
```

## 线路与门

线路来自 QuantumCircuits 的 `Circuit`：`push!` 追加指令，
门为单例或构造函数（`H`、`X`、`CX`、`RX`、`RZZ`、…），
可用 `ctrl` / `pow` / `inv` 修饰；参数门用符号（`:θ`）表示待绑定参数。

```@example
using QuantumCircuits, VQC

c = Circuit(3)
push!(c, H(1))                   # 单比特门
push!(c, CX(1, 2))               # 受控门
push!(c, RZZ(0.3, 2, 3))         # 常数值参数门
push!(c, RX(:θ, 3))              # 符号参数门
push!(c, ctrl(H, 1)(1, 2))       # ctrl 修饰：qubit 1 控制 H 作用在 qubit 2
push!(c, barrier(1, 2, 3))       # 屏障
depth(c), count_ops(c)
```

## 演化线路

```@docs
simulate
```

```@example
using QuantumCircuits, VQC

c = Circuit(2)
push!(c, H(1))
push!(c, CX(1, 2))

ψ = simulate(c, zero_state(2))       # 非就地：返回新态
x = zero_state(2)
simulate!(c, x)                      # 就地
probabilities(ψ)
```

符号参数在演化时绑定（`params` 接受按序向量或 `Dict`）：

```@example
using QuantumCircuits, VQC

cp = Circuit(2)
push!(cp, RX(:θ1, 1))
push!(cp, RY(:θ2, 2))

s1 = simulate(cp, zero_state(2); params = [0.4, -0.2])          # 按 parameters(c) 顺序
s2 = simulate(cp, zero_state(2); params = Dict(:θ1 => 0.4, :θ2 => -0.2))
s1 ≈ s2
```

`params` 传 `Dict` 时键可以是 `Param` / `Symbol`。
同名参数共享（权重共享）；也可以先用 QuantumCircuits 的
`assign` / `assign!` 把符号替换为数值再演化。

## 直接作用局域矩阵与 Kraus 算子

绕开 IR，直接把局域矩阵 / Kraus 算子集**就地**作用到态上
（核心层原语 `apply!` / `apply_kraus!`，`positions[1]` 为矩阵最高位；
详见[核心层 API](@ref)）：

```@example
using VQC, LinearAlgebra

ψ = rand_state(3)
apply!(ψ, Matrix{Float64}(I, 2, 2), (1,))   # 单位阵作用在 qubit 1（就地）
ρ = apply_kraus!(DensityMatrix(ψ), [[1.0 0.0; 0.0 0.0], [0.0 0.0; 0.0 1.0]], (1,))
```

## 测量与后选择

```@example
using QuantumCircuits, VQC

c = Circuit(2)
push!(c, H(1))
push!(c, CX(1, 2))
ψ = simulate(c, zero_state(2))

probabilities(ψ, 1)              # 单比特分布
outcome, p = measure!(ψ, 1)      # 测量并坍缩
samples = sample(ψ, 16)          # 计算基采样
```

后选择用 `post_select!` / `post_select`；偏迹 `partial_tr`、
保真度 `fidelity` 等信息量函数见[核心层 API](@ref)。
自旋算符代数（`SpinOpTerm` / `SpinOpSum`）与时间演化见
[哈密顿量与自旋算符](@ref)。
