# 用变分量子算法求海森堡链基态

本教程演示 **VQA（Variational Quantum Algorithm）** 求解一维海森堡链
基态能量的完整流程，全部基于本包栈：

* **哈密顿量**：用 `QuantumCircuits.Hamiltonian` 的 Pauli 代数构造（两体耦合项）；
* **ansatz**：`QuantumCircuits.variational_circuit_1d`（hardware-efficient，
  `RZ-RY-RZ` + 最近邻 `CX` 纠缠层）；
* **期望值与演化**：VQC 后端（`simulate` + `VQC.expectation`，走局域 kernel，不构造稠密矩阵）；
* **梯度**：`VQCZygoteExt` 提供的端到端自动微分——对线路参数向量求
  `∂⟨ψ(θ)|H|ψ(θ)⟩/∂θ`，单次反向约等于 2–3 次前向的代价。

运行本教程需要环境里同时加载 `QuantumCircuits`、`VQC` 与 `Zygote`：

```julia
using Pkg
Pkg.add("Zygote")          # 触发 VQCZygoteExt
```

## 1. 模型：一维开链海森堡

取长度 `L` 的一维自旋链，最近邻反铁磁耦合 `J = 1`：

```math
H = J \sum_{i=1}^{L-1}\left(X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1}\right)
```

每一项都是两体 Pauli 串，可直接用 `PauliTerm` 表达，汇总为 `PauliSum`：

```julia
using QuantumCircuits
using QuantumCircuits.Hamiltonian: PauliTerm, PauliSum
using VQC
using Zygote, LinearAlgebra, Random, Printf

"一维开链海森堡模型 H = Σᵢ (XᵢXᵢ₊₁ + YᵢYᵢ₊₁ + ZᵢZᵢ₊₁)"
function heisenberg_chain(L::Int; J::Real = 1.0)
    terms = PauliTerm[]
    for i in 1:L-1
        push!(terms, PauliTerm(J, i => :X, i + 1 => :X))
        push!(terms, PauliTerm(J, i => :Y, i + 1 => :Y))
        push!(terms, PauliTerm(J, i => :Z, i + 1 => :Z))
    end
    return PauliSum(terms)
end

L = 4
H = heisenberg_chain(L)
```

小规模下可以展开成稠密矩阵求精确基态能量，作为验证基准
（`QuantumCircuits.mat`，qubit 1 为最低有效位）：

```julia
using QuantumCircuits: mat, parameters

M = Matrix(mat(H, L))
E0 = minimum(real.(eigvals(Hermitian(M))))      # L = 4 时 ≈ -6.4641016…
```

## 2. ansatz：一维变分线路

用包内现成的 hardware-efficient ansatz（`depth + 1` 个 `RZ-RY-RZ` 旋转层、
层间 `CX` 梳，奇数层正序 / 偶数层倒序）：

```julia
depth = 3
c = variational_circuit_1d(L, depth)
npar = length(parameters(c))                    # 3L(depth+1) = 48（L=4, depth=3）
```

线路带 `3L(depth+1)` 个**符号参数** `θ[1], …, θ[n]`；给 `simulate` 传数值向量
`θ` 时按 `parameters(c)` 的顺序绑定。

## 3. 损失函数与自动微分梯度

变分原理给出损失 = 能量期望：

```julia
energy(θ) = real(VQC.expectation(H, simulate(c, zero_state(L); params = θ)))
```

> 名字说明：`QuantumCircuits` 与 `VQC` 都导出 `expectation`，这里用
> `VQC.expectation` 限定到 VQC 在 `StateVector` 上的局域实现——它同时也是
> `VQCZygoteExt` 里注册了伴随方法的那一个（对 `PauliTerm` / `PauliSum`）。

`simulate(c, s; params = θ)` 传**向量**时整条模拟链路对 `θ` 可微
（传 `Dict` 则不可微）：梯度反向传播时先正向重放记录每条门的输入态，
再按逆序把期望的态余切经各参数门的雅可比拉回，同名参数梯度自动累加。

用 `Zygote.gradient` 即可得到整条链路的解析梯度：

```julia
θ0 = randn(npar) .* 0.5
g = Zygote.gradient(energy, θ0)[1]      # ∂⟨ψ(θ)|H|ψ(θ)⟩/∂θ
```

梯度正确性可以用中心差分抽查验证（与 AD 偏差应为 ~1e-9），
完整写法见第 4 节脚本；那里还包含 Adam 优化循环。

## 4. Adam 优化循环（完整可运行代码）

把 1–3 节拼起来，用 Adam 做 400 步梯度下降。完整脚本如下，
保存为 `heisenberg_vqa.jl` 后 `julia heisenberg_vqa.jl` 即可运行：

```julia
using QuantumCircuits
using QuantumCircuits.Hamiltonian: PauliTerm, PauliSum
using QuantumCircuits: mat, parameters
using VQC
using Zygote, LinearAlgebra, Random, Printf

function heisenberg_chain(L::Int; J::Real = 1.0)
    terms = PauliTerm[]
    for i in 1:L-1
        push!(terms, PauliTerm(J, i => :X, i + 1 => :X))
        push!(terms, PauliTerm(J, i => :Y, i + 1 => :Y))
        push!(terms, PauliTerm(J, i => :Z, i + 1 => :Z))
    end
    return PauliSum(terms)
end

L = 4
H = heisenberg_chain(L)
E0 = minimum(real.(eigvals(Hermitian(Matrix(mat(H, L))))))
@printf("精确基态能量 E0 = %.10f\n", E0)

depth = 3
c = variational_circuit_1d(L, depth)
npar = length(parameters(c))
println("参数个数 = ", npar)

energy(θ) = real(VQC.expectation(H, simulate(c, zero_state(L); params = θ)))

# ── 梯度正确性：自动微分 vs 中心差分（前 3 个参数） ──
rng = MersenneTwister(1)
θ0 = randn(rng, npar) .* 0.5
g = Zygote.gradient(energy, θ0)[1]
fd = zeros(npar)
hh = 1e-6
for j in 1:3
    θp = copy(θ0); θp[j] += hh
    θm = copy(θ0); θm[j] -= hh
    fd[j] = (energy(θp) - energy(θm)) / (2hh)
end
gmax = maximum(abs.(g[1:3] .- fd[1:3]))
println("梯度校验: AD = ", round.(g[1:3]; digits = 8),
        "  FD = ", round.(fd[1:3]; digits = 8),
        "  最大偏差 = ", round(gmax; sigdigits = 2))

# ── Adam 优化 ──
θ = copy(θ0)
m = zeros(npar); v = zeros(npar)
β1, β2, η = 0.9, 0.999, 0.08
Ehist = Float64[]
for it in 1:400
    E, back = Zygote.pullback(energy, θ)
    g = back(1.0)[1]
    m .= β1 .* m .+ (1 - β1) .* g
    v .= β2 .* v .+ (1 - β2) .* g.^2
    θ .-= η .* (m ./ (1 - β1^it)) ./ (sqrt.(v ./ (1 - β2^it)) .+ 1e-8)
    push!(Ehist, E)
    (it % 50 == 0 || it == 1) &&
        @printf("  it=%3d  E = %.10f   (E-E0 = %.3e)\n", it, E, E - E0)
end
@printf("收敛误差 E-E0 = %.3e\n", Ehist[end] - E0)
```

### 运行输出（seed 固定，L = 4, depth = 3, 400 步）

```text
精确基态能量 E0 = -6.4641016151
参数个数 = 48
梯度校验: AD = [-0.0, -0.08200633, 0.01046681]  FD = [0.0, -0.08200633, 0.01046681]  最大偏差 = 9.7e-10
  it=  1  E = 1.9082560442   (E-E0 = 8.372e+00)
  it= 50  E = -6.3801443455   (E-E0 = 8.396e-02)
  it=100  E = -6.4610668756   (E-E0 = 3.035e-03)
  it=150  E = -6.4637964955   (E-E0 = 3.051e-04)
  it=200  E = -6.4640550440   (E-E0 = 4.657e-05)
  it=250  E = -6.4640755863   (E-E0 = 2.603e-05)
  it=300  E = -6.4640782670   (E-E0 = 2.335e-05)
  it=350  E = -6.4640801838   (E-E0 = 2.143e-05)
  it=400  E = -6.4640820121   (E-E0 = 1.960e-05)
收敛误差 E-E0 = 1.960e-05
```

几点观察：

1. **自动微分是可靠的**：`Zygote.gradient` 与中心差分在前 3 个参数上偏差 ~1e-9，
   说明 `VQCZygoteExt` 沿 `simulate → expectation` 的回传链路正确；
2. **优化稳定收敛**：Adam 在 400 步内把 `L=4` 的能量从初值 ~1.9 压到
   `E0` 之上 ~2e-5——已经远小于基态以上第一条激发能隙（≈ 1.8），
   可认为求得了基态；
3. 每一步 `Zygote.pullback` 只比一次 `energy(θ)` 前向贵约 2–3 倍。

## 5. 更大的系统怎么办

`L = 6` 时同样用 Adam 优化（3 组随机初值取最优）：`depth=3` 只能到
`E−E0 ≈ 0.12`，加深到 `depth=6` 后到 `E−E0 ≈ 1.5e-2`。这是**通用 HEA
的典型局限**，不是 AD/后端的问题：

* 加深纠缠层、用小学习率 + 退火、或做多次随机重启都能改善；
* 更有效的是利用物理对称性（例如针对自旋单态构造**对称保持 ansatz**，
  或对链长做块化），可把参数空间压缩到基态所在的对称扇区。

## 相关页面

* 页《哈密顿量与自旋算符》（`ham.md`）：`SpinOpSum` / `PauliSum` 的构造、`expectation` 与局域作用；
* 页《变分量子线路》（`variational.md`）：`simulate(..., params=θ)` 向量绑参的机制与限制。
