# 哈密顿量与自旋算符

VQC 核心层提供自旋算符代数 `SpinOpTerm` / `SpinOpSum`
（QuantumCircuits `PauliTerm` / `PauliSum` 的推广）：
单个比特位上可以放 `Symbol` 简写（`:I / :X / :Y / :Z / :P / :M`）
或**任意 `2×2` 矩阵**（不限 Pauli / 厄米）。

## 构造与代数

```@example
using VQC

t = SpinOpTerm(0.5, 1 => :Z, 2 => :Y)       # 0.5 · Z₁ Y₂
A = ComplexF64[0.5 0.2im; -0.3 1.2]         # 任意 2×2 矩阵
t2 = SpinOpTerm(1.0, 3 => A)

ham = SpinOpSum([SpinOpTerm(0.5, 1 => :Z, 2 => :Z),
                 SpinOpTerm(1.0, 2 => :X),
                 t2])                        # 0.5·Z₁Z₂ + X₂ + A₃

t + t2                                       # SpinOpTerm 相加 → SpinOpSum
2 * ham                                      # 数乘
adjoint(ham)                                 # 厄米共轭
```

```@docs
SpinOpTerm
SpinOpSum
```

## 展开为矩阵（小规模验证用）

`VQC.mat(ham, n)` 把 `SpinOpTerm` / `SpinOpSum` 展开为 `n` 比特的
`2ⁿ × 2ⁿ` 稠密矩阵（`mat` 未导出，用 `VQC.mat` 引用），
仅供小规模验证 / 对拍；大规模请走 `apply` 的局域核路径。

```@example
using VQC

const mat = VQC.mat                          # mat 未导出

ham = SpinOpSum([SpinOpTerm(0.5, 1 => :Z, 2 => :Z),
                 SpinOpTerm(1.0, 2 => :X)])
M = mat(ham, 2)                              # 4×4 稠密矩阵
```

## 高效作用（无需构造大矩阵）

`apply` 把算符逐因子经局域 kernel 作用到态上，
复杂度 `O(#ops · 2ⁿ)`，适合**本征态求解**（Lanczos 的作用算符）
与**时间演化**（Trotter 步）：

```@docs
apply(t::SpinOpTerm, s::StateVector)
```

```@example
using VQC, LinearAlgebra

const mat = VQC.mat

n = 3
ψ = rand_state(n)
ham = SpinOpSum([SpinOpTerm(0.5, 1 => :Z, 2 => :Z),
                 SpinOpTerm(1.0, 2 => :X, 3 => :Y)])
hamψ = apply(ham, ψ)                         # H|ψ⟩，不构造 8×8 矩阵
norm(storage(hamψ) - mat(ham, n) * storage(ψ))   # 与稠密矩阵一致
```

时间演化：一阶 Trotter 分解 `exp(-iΔt H) ≈ Πₖ exp(-iΔt Hₖ)`，
每个 `exp(-iΔt Hₖ)` 逐项（必要时再分解到单因子）经 `apply` 作用：

```@example
using VQC, LinearAlgebra

const mat = VQC.mat

n, Δt = 2, 0.1
ham = SpinOpSum([SpinOpTerm(1.0, 1 => :Z, 2 => :Z), SpinOpTerm(1.0, 2 => :X)])
ψ = rand_state(n)

"单步一阶 Trotter：逐项 exp(-iΔt·term) 依次作用"
function trotter_step(ham, v, n, Δt)
    for term in ham.terms
        v = exp(-1.0im * Δt * mat(term, n)) * v
    end
    return v
end

ψt = trotter_step(ham, storage(ψ), n, Δt)
norm(ψt - exp(-1.0im * Δt * mat(ham, n)) * storage(ψ)) < 1e-2
```

## 期望值

```@docs
expectation
```

```@example
using VQC

ψ = rand_state(2)
ham = SpinOpSum([SpinOpTerm(0.5, 1 => :Z, 2 => :Z),
                 SpinOpTerm(1.0, 2 => :X)])
expectation(ham, ψ)                          # ⟨ψ|H|ψ⟩（走局域核路径）
ρ = DensityMatrix(ψ)
expectation(ham, ρ)                          # tr(ρH)
```

厄米算符的期望值为实数；`LinearAlgebra.ishermitian` 可用于检验。
与 QuantumCircuits 的 Pauli 代数（`PauliTerm` / `PauliSum`）互操作时，
`expectation` 同样由 VQC 后端实现。
