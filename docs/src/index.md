# VQC.jl

VQC 是 [QuantumCircuits](https://github.com/) IR 的**态矢量 / 密度矩阵模拟后端**：
把 `Circuit` 中的指令（`GateOp` / `ChannelOp` / `MeasOp` / `IfOp` / …）高效地
作用到量子态上，并提供测量、偏迹、期望值与自旋算符代数等原语。
自动微分由包扩展 `VQCZygoteExt` 提供（`using Zygote` + `using QuantumCircuits` 时自动加载）。

约定（与 QuantumCircuits 一致）：比特索引 **1-based**、小端序
（qubit 1 = 最低有效位）；门矩阵的第一个比特为矩阵最高位。

## 一个简单的例子：Bell 态

```@example
using QuantumCircuits, VQC

c = Circuit(2)
push!(c, H(1))
push!(c, CX(1, 2))

ψ = simulate(c, zero_state(2))
println("probabilities = ", probabilities(ψ))

outcome, p = measure!(ψ, 1)
println("outcome = $outcome, probability = $p")
```

## 变分线路

线路中的参数门可以用符号（`:θ`）构造，演化时通过 `params` 绑定数值：

```@example
using QuantumCircuits, VQC

c = Circuit(2)
push!(c, RX(:θ, 1))
push!(c, RY(:θ, 2))
push!(c, CX(1, 2))

println("symbolic parameters: ", parameters(c))

ψ = simulate(c, zero_state(2); params = [0.4])
println("probabilities = ", probabilities(ψ))
```

配合 Zygote 可以对参数求梯度（详见[变分量子线路](@ref)）：

```julia
using Zygote

target = rand_state(2)
loss(θ) = distance(target, simulate(c, zero_state(2); params = θ))
θ0 = randn(num_params(c))
grad = Zygote.gradient(loss, θ0)[1]
```

## 目录

```@contents
Depth = 2
```
