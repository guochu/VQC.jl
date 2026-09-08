# 核心层 API

VQC 的核心层**不依赖任何 IR 包**：只 `using VQC` 即可完成量子态构造、
局域矩阵 / Kraus 作用、测量、偏迹、信息量计算与自旋算符代数——
这些接口不涉及 `Circuit` / `GateOp`，可与任何自定义流程组合。
与 QuantumCircuits 线路的协同（`simulate` / 参数绑定 / 自动微分）
见[快速开始](@ref)与[变分量子线路](@ref)；比特与索引约定见
[快速开始](@ref)首节。

## 量子态：构造与存储

```@docs
StateVector
DensityMatrix
zero_state
rand_state
rand_densitymatrix
```

```@example
using VQC

ψ = rand_state(4)                  # 随机纯态（2⁴ 个复振幅）
ρ = DensityMatrix(ψ)               # 纯态 → 密度矩阵
nq = 3
amp = amplitude_encoding([1.0, 2.0im, 0.5]; nqubits = 2)   # 振幅编码（自动归一化）
nothing
```

## 态编码

```@docs
onehot_encoding
qubit_encoding
amplitude_encoding
```

```@example
using VQC

onehot_encoding([1, 0])            # |q₂q₁⟩ = |01⟩：qubit 1 = 1
qubit_encoding([π/2, 0.0])         # qubit 1 = |+⟩ 方向，qubit 2 = |0⟩
amplitude_encoding([1.0, 2.0im]; nqubits = 1)
```

## 读取振幅与比特置换

```@docs
amplitude
amplitudes
permute
```

```@example
using VQC

ψ = rand_state(3)
amplitude(ψ, [1, 0, 1])            # ⟨101|ψ⟩（bits[i] = qubit i 的取值）
permute(ψ, [2, 3, 1])              # 新态的 qubit 1 = 原 qubit 2，以此类推
```

## 就地重置

```@docs
reset!
reset_onehot!
reset_qubit!
```

## 局域作用原语（就地）

局域矩阵与 Kraus 算子集**就地**作用到态上（`positions` 用
`NTuple` 传入，`positions[1]` 为矩阵最高位）：

```@docs
apply!
apply_kraus!
reset_qubit_zero!
```

```@example
using VQC, LinearAlgebra

ψ = rand_state(3)
apply!(ψ, [0 1; 1 0], (2,))        # X 作用在 qubit 2（就地）
ρ = DensityMatrix(ψ)
apply_kraus!(ρ, [[1.0 0.0; 0.0 0.0], [0.0 0.0; 0.0 1.0]], (1,))   # 退相位
```

## 测量、采样与后选择

```@docs
probabilities
marginal_probabilities
measure!
sample
post_select!
post_select
```

```@example
using VQC

ψ = rand_state(3)
marg = marginal_probabilities(ψ, [2, 1])   # qubit 2、1 的联合分布（第 j 位对应 qubits[j]）
outcome = measure!(ψ, 3)           # 测量 qubit 3 并坍缩（就地），返回 0/1
dict = sample(ψ, 100)              # 计算基采样：基矢索引 → 次数
o_all = measure!(ψ)                # 测量全部比特，返回小端序整数
nothing
```

非就地版本（返回新态）：`measure(s, q)`、`post_select(s, q, val)`。

## 可观测量与偏迹

```@docs
partial_tr
```

一般矩阵的期望值 `expectation(M, state)`（以及自旋算符
`expectation(ham, state)`）见[哈密顿量与自旋算符](@ref)。

```@example
using VQC, LinearAlgebra

ψ = rand_state(4)
ρ = partial_tr(ψ, [1, 2])          # 约化到 qubit 3、4（保留比特升序重编号）
expectation(Matrix{Float64}(I, 4, 4), ρ)   # tr(ρ) = 1
```

## 保真度、距离与信息量

```@docs
fidelity
distance
distance2
schmidt_numbers
entropy
renyi_entropy
```

```@example
using VQC

ψ, φ = rand_state(3), rand_state(3)
ρ = DensityMatrix(ψ)
fidelity(ψ, φ)                     # |⟨ψ|φ⟩|²
entropy(ρ)                         # 谱（香农）熵
renyi_entropy(partial_tr(ψ, [1]))  # 单比特约化态的 Rényi-2 熵
nothing
```

自旋算符代数（`SpinOpTerm` / `SpinOpSum`，支持任意 2×2 局域算子）
与高效 `apply` / 期望值见[哈密顿量与自旋算符](@ref)。
