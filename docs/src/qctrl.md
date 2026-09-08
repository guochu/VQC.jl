# 测量与经典反馈

`simulate` 在演化线路时维护**经典寄存器**运行时：`MeasOp` 把测量
结果写回经典位，`IfOp` 按经典位的值条件执行子线路。这可以用来表达
含中途测量与前馈控制的动态量子线路。

## 中途测量 + 条件分支

下面的线路：对 qubit 1 施加 `H` 后测量并写入经典位 `c`；
若 `c == 1`（测得 |1⟩）则再施加 `X` 把态翻回 |0⟩：

```@example
using QuantumCircuits, VQC

c = Circuit(1)
push!(c, H(1))
push!(c, measure(1, c.cregs[1][1]))                    # 测量 → 经典位
QuantumCircuits.if_then(c, c.cregs[1][1] == 1, Circuit([X(1)]))

ψ = simulate(c, zero_state(1))
probabilities(ψ)                                        # 总是被翻回 |0⟩
```

无条件分支时，测量坍缩直接体现在态上；测量结果可通过经典位
在后续 `IfOp` 中使用。

## 线路内的测量指令

`apply!` 也支持把单条测量指令直接作用到态上（坍缩，经典结果写在
经典位对象里）：

```@example
using QuantumCircuits, VQC

c = Circuit(1)                              # 经典寄存器随线路创建
push!(c, H(1))
ψ = zero_state(1)
apply!(ψ, H(1))
apply!(ψ, measure(1, c.cregs[1][1]))        # 测量并坍缩（结果写入经典位）
probabilities(ψ)                            # 测后只剩 |0⟩ 或 |1⟩ 之一
```

## 纯态与混合态

* 中途无测量的幺正线路：`StateVector` 与 `DensityMatrix` 均可；
* 一旦出现测量 / 噪声信道：
  * `StateVector` 路径直接坍缩（测量是随机的，结果不可重复）；
  * 如需对"所有测量分支"求平均（描述系综），先把态提升为
    `DensityMatrix`（噪声信道会自动完成提升）。

## 采样统计

对态逐比特 `measure!` 做多次采样统计（含经典反馈的线路直接循环 `simulate`）：

```@example
using QuantumCircuits, VQC

c = Circuit(2)
push!(c, H(1))
push!(c, CX(1, 2))

counts = Dict{Int,Int}()
for _ in 1:200
    ψr = simulate(c, zero_state(2))
    b2 = measure!(ψr, 2)
    b1 = measure!(ψr, 1)
    key = b2 << 1 | b1                     # (q2, q1) → 整数
    counts[key] = get(counts, key, 0) + 1
end
counts                                      # Bell 态：00 与 11（=3）各约 50%
```

## 完整示例：量子隐形传态

把 qubit 1 上的未知态 `|ψ⟩` 传送到 qubit 3：qubit 2、3 纠缠作量子通道，
测量 qubit 1、2 后按两个经典位的值对 qubit 3 做 `X` / `Z` 前馈修正。
这是测量 + 经典控制语句的最小完整算法：

```@example teleport
using QuantumCircuits, VQC

c = Circuit(3)
creg = c.cregs[1]
push!(c, H(1))                    # 待传态：q1 = |+⟩ = H|0⟩
push!(c, H(2))                    # Bell 对准备：H q2
push!(c, CX(2, 3))                # CX q2→q3
push!(c, CX(1, 2))                # CX q1→q2（Bell 基测量准备）
push!(c, H(1))
push!(c, measure([1, 2], [creg[1], creg[2]]))   # 测量写入经典位
QuantumCircuits.if_then(c, creg[2] == 1, Circuit([X(3)]))   # 位修正
QuantumCircuits.if_then(c, creg[1] == 1, Circuit([Z(3)]))   # 相位修正

function teleport_run()
    simulate(c, zero_state(3))    # 每次运行：随机测量 + 经典反馈
end

c                                 # 渲染完整线路
```

验证：无论经典位取何值，修正后 qubit 3 都应处于 `|+⟩`——
对 qubit 1、2 求偏迹后与 `|+⟩` 的保真度恒为 1：

```@example teleport
ρ3 = partial_tr(teleport_run(), [1, 2])          # qubit 3 的约化态
fidelity(ρ3, qubit_encoding([0.5]))              # |+⟩（θ = 0.5）的保真度 ≈ 1
```

两个 `if_then` 分支按经典位独立触发——四次测量结果
`(b₁, b₂) ∈ {00, 01, 10, 11}` 分别对应"无修正 / X / Z / XZ"，
每次运行只会命中其一。
