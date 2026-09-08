# 变分量子线路

变分量子线路 = 含**符号参数门**的 `Circuit` + 外层经典优化器。
VQC 通过 `simulate(c, ψ; params = θv)` 的向量绑定入口支持
对参数的端到端自动微分（`VQCZygoteExt`）。

## 构造变分线路

参数门直接用符号构造：`RX(:θ, 1)`。同名参数自动共享。

```@example
using QuantumCircuits, VQC

function variational_circuit(L::Int, depth::Int)
    c = Circuit(L)
    for i in 1:L
        push!(c, RZ(:θ, i))
        push!(c, RY(:θ, i))
    end
    for _ in 1:depth
        for i in 1:L-1
            push!(c, CX(i, i + 1))
        end
        for i in 1:L
            push!(c, RZ(:θ, i))
            push!(c, RX(:θ, i))
        end
    end
    return c
end

c = variational_circuit(3, 2)
ps = parameters(c)              # 线路的符号参数（按序，同名去重）
length(ps)
```

参数可以随时通过 `params` kwarg 绑定，也可以用 QuantumCircuits 的
`assign!` 把数值写进线路副本：

```@example
using QuantumCircuits, VQC

c = Circuit(1)
push!(c, RX(:θ, 1))
θv = [0.7]
ψ = simulate(c, zero_state(1); params = θv)
c2 = QuantumCircuits.assign(copy(c), Dict(:θ => 0.7))
simulate(c2, zero_state(1)) ≈ ψ
```

## 梯度与优化

`params` 传**向量**时整条模拟链路对 `θv` 可微；传 `Dict` 时参数梯度不可用。

```julia
using Pkg
Pkg.add("Zygote")

using QuantumCircuits, VQC, Zygote

c = variational_circuit(3, 2)              # 上一节的构造
target = rand_state(3)

loss(θ) = distance(target, simulate(c, zero_state(3); params = θ))

θ0 = randn(length(parameters(c)))
loss(θ0)                        # 初值损失
grad = Zygote.gradient(loss, θ0)[1]   # ∂loss/∂θ
```

配合 Optimisers.jl / Flux 的优化器做梯度下降：

```julia
using Flux.Optimise

opt = ADAM(0.1)
θ = copy(θ0)
for i in 1:100
    g = Zygote.gradient(loss, θ)[1]
    Optimise.update!(opt, θ, g)
    println("epoch $i: loss = $(loss(θ))")
end
```

## 注意事项

* 含测量（`MeasOp`）或 `IfOp` 的线路不可微，`Zygote.gradient` 会给出明确错误；
* 含噪线路（`ChannelOp`）须演化到 `DensityMatrix`，参数梯度同样经由
  `params = θv` 向量入口回传；
* 大规模线路上梯度开销约为前向的 2–3 倍（checkerboard 反向传播）。
