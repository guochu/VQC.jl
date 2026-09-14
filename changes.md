# 接口变更记录（changes.md）

## v0.2 — 保真度接口：混合态-混合态改为具名约定（2026-09）

### 动机

保真度在**纯态-纯态**（`|⟨ψ|φ⟩|²`）与**纯态-混合态**（`⟨ψ|ρ|ψ⟩`）情形
各文献约定一致、无歧义；但**混合态-混合态**存在两种标准约定
（平方 / 开方），不能共用一个名字。旧版 `fidelity(ρ::DM, σ::DM)`
实现为 `tr(√ρ √σ)` —— 该量不属于任何标准约定（仅当两态可交换或
一侧为纯态时才与约定值重合），且与纯态情形的数值衔接存在隐患。

### 变更内容

| 场景 | 旧 | 新 |
|---|---|---|
| 纯态-纯态 | `fidelity = \|⟨ψ\|φ⟩\|²` | **不变** |
| 纯态-混合态 | `fidelity = ⟨ψ\|ρ\|ψ⟩` | **不变** |
| 混合态-混合态 | `fidelity = tr(√ρ √σ)`（非标准） | **移除该方法**，改用具名函数： |

新增两个具名定义（均要求两侧为 `DensityMatrix`）：

```julia
fidelity_squared(ρ, σ)   # (tr√(√ρ σ √ρ))²   平方约定（Nielsen–Chuang / Qiskit state_fidelity）
fidelity_root(ρ, σ)      #  tr√(√ρ σ √ρ)      开方约定（root fidelity），= √F
```

任一侧为纯态时 `fidelity_squared` 与 `fidelity` 数值一致（可无缝迁移）。

> 数学附注：`tr(√ρ √σ)` 是两个矩阵开方乘积的**迹**（特征值之和），
> 而 Uhlmann 的 `tr√(√ρ σ √ρ)` 是乘积的**奇异值之和**（核范数），
> 二者一般不相等（等号当且仅当 ρ、σ 可交换）。因此旧实现即使在
> "开方约定"下也是错的。

### 迁移指南

```julia
# 旧（非标准量，已不可用）
f = fidelity(ρ, σ)            # ρ、σ 均为 DensityMatrix

# 新：按所需约定二选一
F = fidelity_squared(ρ, σ)    # Nielsen–Chuang / Qiskit 口径（推荐）
f = fidelity_root(ρ, σ)       # 开方口径，F = f²

# 注意：旧返回值 ≠ 二者中的任何一个；若旧代码依赖 tr(√ρ√σ) 的数值，
# 需显式改写为 tr(sqrt(ρ) * sqrt(σ))（不推荐）或改用标准约定。
```

涉及纯态的调用（`fidelity(ψ, φ)` / `fidelity(ρ, ψ)` / `fidelity(ψ, ρ)`）
**无需改动**。

### 涉及文件

- `src/states/densitymatrix.jl`（移除 DM-DM `fidelity`，新增两个具名函数）
- `src/states/statevector.jl`（`fidelity` docstring 更新与引导）
- `src/VQC.jl`（导出 `fidelity_squared` / `fidelity_root`）
- `test/states.jl`（约定一致性测试：`root² == squared`、纯态一侧与 `fidelity` 一致）
- `docs/src/coreapi.md`（`@docs` 块与示例同步）

全量测试 179/179 通过。
