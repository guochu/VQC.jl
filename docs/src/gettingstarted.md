# 快速开始

本节给出用 VQC 做量子线路模拟的基本流程：构造态 → 构造线路 → 演化 → 测量。

## 量子态

态矢量为 `StateVector`（纯态），密度矩阵为 `DensityMatrix`（混合态）。
二者内部存储可通过 `storage` 取出（`StateVector` 为长度 `2^n` 的向量，
`DensityMatrix` 为长度 `4^n` 的平坦向量，行索引为低位）。

```@docs
StateVector
DensityMatrix
zero_state
rand_state
rand_densitymatrix
```

`qubit_encoding` / `amplitude_encoding` / `onehot_encoding` 从经典数据构造态；
`amplitude` / `amplitudes` 读取振幅；`permute` 置换比特轴。

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
apply!
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
（核心层原语 `apply!` / `apply_kraus!`，`positions[1]` 为矩阵最高位）：

```@docs
apply_kraus!
reset_qubit_zero!
```

```@example
using VQC, LinearAlgebra

ψ = rand_state(3)
apply!(ψ, Matrix{Float64}(I, 2, 2), [1])   # 单位阵作用在 qubit 1（就地）
ρ = apply_kraus!(DensityMatrix(ψ), [[1.0 0.0; 0.0 0.0], [0.0 0.0; 0.0 1.0]], [1])
```

## 测量与后选择

```@docs
probabilities
measure!
measure
sample
post_select!
post_select
```

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

## 可观测量与约化

```@docs
partial_tr
fidelity
```

自旋算符代数（`SpinOpTerm` / `SpinOpSum`）与时间演化见
[哈密顿量与自旋算符](@ref)。
