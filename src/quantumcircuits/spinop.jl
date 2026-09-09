# spinop.jl — SpinOpTerm / SpinOpSum 的状态作用与期望值
#
# （SpinOp 类型定义已上移至 QuantumCircuits.Hamiltonian；本文件只提供
#   依赖 VQC 态类型与局域 kernel 的方法：apply / mul! / _axpy! / expectation。）

using QuantumCircuits.Hamiltonian: SpinOpTerm, SpinOpSum, _spin_matrix
import LinearAlgebra: mul!, rmul!, dot

# ── 高效作用：apply(SpinOpTerm, StateVector) ────────────────────────────────

"""
    apply(t::SpinOpTerm, s::StateVector) -> StateVector

把自旋算符作用到态矢量上（**非就地**，返回新态）；
`SpinOpSum` 的对应方法 `apply(s::SpinOpSum, s0)` 为各项作用的线性组合。

乘积项 `t = c · A₁ A₂ …` 无需构造大矩阵：逐因子走局域 kernel，
复杂度 `O(#ops · 2ⁿ · D_local)`。
适用于高效本征态求解（Lanczos / 精确对角化的作用算符）与
时间演化（Trotter 步：`exp(-i·Δt·H) ≈ Πₖ apply(termₖ, ·)`）。

实数态遇复算子自动提升为复数版本。
"""
function apply(t::SpinOpTerm, s::StateVector)
    v = copy(storage(s))
    n = _nqubits(s)
    for (q, op) in reverse(t.ops)      # 逆序作用 = 正序矩阵积 A₁ A₂ ⋯
        m = _spin_matrix(op)
        1 <= q <= n || throw(ArgumentError("operator position $q out of range [1, $n]"))
        if eltype(v) <: Real && !(eltype(m) <: Real)
            v = convert(Vector{complex(float(eltype(v)))}, v)
        end
        apply_kernel!(v, (q - 1,), m)
    end
    rmul!(v, t.coeff)
    return StateVector(v, _nqubits(s))
end

# ── LinearAlgebra.mul! / axpy!：SpinOp 的无中间态作用 ────────────────────────
#
#   mul!(y, t, x)            y = t·x（公开接口，LinearAlgebra 惯例）
#   _axpy!(α, t, x, y, ws)   y += α·(t·x)（内部接口，ws 为工作缓冲）
#
# 单因子项逐块 out-of-place kernel，全程不分配中间振幅数组；多因子项
# 首因子 out-of-place 写入目标 / 工作缓冲，其余因子就地作用，同样零分配。

"实输出态上作用复算子 / 复系数 → 明确报错（避免 kernel 静默 InexactError）。"
function _check_real_output(y, t::SpinOpTerm, α::Number = 1)
    eltype(y) <: Real || return nothing
    (isreal(α) && isreal(t.coeff)) ||
        throw(ArgumentError("complex scalar on real output; provide a complex state"))
    for (_, op) in t.ops
        eltype(_spin_matrix(op)) <: Real ||
            throw(ArgumentError("complex operator on real output; provide a complex state"))
    end
    return nothing
end

# 乘积项的逆序 (key, matrix) 因子表：先作用最右因子 = 正序矩阵积
function _spin_factors(t::SpinOpTerm, n::Int)
    fs = Tuple{NTuple{1,Int},AbstractMatrix}[]
    for (q, op) in reverse(t.ops)
        1 <= q <= n || throw(ArgumentError("operator position $q out of range [1, $n]"))
        push!(fs, ((q - 1,), _spin_matrix(op)))
    end
    return fs
end

function mul!(y::StateVector, t::SpinOpTerm, x::StateVector)
    n = _nqubits(y)
    (_nqubits(x) == n && length(y) == length(x)) ||
        throw(DimensionMismatch("state dimension mismatch"))
    y === x && throw(ArgumentError("output must not alias input"))
    _check_real_output(y, t)
    factors = _spin_factors(t, n)
    key, m = factors[1]
    lmul_kernel!(y.data, x.data, key, m)       # 首因子 out-of-place（y 旧值可覆盖）
    for i in 2:length(factors)                 # 其余因子就地作用
        apply_kernel!(y.data, factors[i]...)
    end
    rmul!(y.data, t.coeff)
    return y
end

function mul!(y::DensityMatrix, t::SpinOpTerm, x::DensityMatrix)
    n = _nqubits(y)
    (_nqubits(x) == n && length(y.data) == length(x.data)) ||
        throw(DimensionMismatch("state dimension mismatch"))
    y.data === x.data && throw(ArgumentError("output must not alias input"))
    _check_real_output(y, t)
    d = 1 << n
    factors = _spin_factors(t, n)
    key, m = factors[1]
    dm_lmul_kernel!(y.data, x.data, d, key, m)
    for i in 2:length(factors)
        apply_kernel!(y.data, factors[i]...)   # 就地行作用
    end
    rmul!(y.data, t.coeff)
    return y
end

"""
    _axpy!(α, t::SpinOpTerm, x, y, ws) -> y    # y += α·(t·x)
    _axpy!(α, s::SpinOpSum, x, y, ws) -> y     # y += Σₖ α·(tₖ·x)

内部接口：就地累加。单因子项逐块 kernel，全程零分配；多因子项把
`t·x` 的前 `k-1` 个因子作用到工作缓冲 `ws`（内容可被覆盖），最后
一个因子以 axpy kernel 直接融合累加进 `y`——无需整向量二次遍历。
`ws` 由调用方创建并复用（仅多因子项需要），内部不再分配量子态。
`x` 与 `y` 不得共享存储。
"""
function _axpy!(α::Number, t::SpinOpTerm, x::StateVector, y::StateVector,
                ws::Union{StateVector,Nothing})
    n = _nqubits(y)
    (_nqubits(x) == n && length(y) == length(x)) ||
        throw(DimensionMismatch("state dimension mismatch"))
    y === x && throw(ArgumentError("output must not alias input"))
    _check_real_output(y, t, α)
    factors = _spin_factors(t, n)
    if length(factors) == 1
        key, m = factors[1]
        axpy_kernel!(y.data, α * t.coeff, x.data, key, m)
        return y
    end
    ws === nothing && throw(ArgumentError("multi-factor term requires a workspace"))
    key, m = factors[1]
    lmul_kernel!(ws.data, x.data, key, m)          # ws ← f₁·x
    for i in 2:length(factors) - 1                 # 中间因子就地作用
        apply_kernel!(ws.data, factors[i]...)
    end
    key, m = factors[end]
    axpy_kernel!(y.data, α * t.coeff, ws.data, key, m)   # 末因子融合累加
    return y
end

function _axpy!(α::Number, t::SpinOpTerm, x::DensityMatrix, y::DensityMatrix,
                ws::Union{DensityMatrix,Nothing})
    n = _nqubits(y)
    (_nqubits(x) == n && length(y.data) == length(x.data)) ||
        throw(DimensionMismatch("state dimension mismatch"))
    y.data === x.data && throw(ArgumentError("output must not alias input"))
    _check_real_output(y, t, α)
    d = 1 << n
    factors = _spin_factors(t, n)
    if length(factors) == 1
        key, m = factors[1]
        dm_axpy_kernel!(y.data, α * t.coeff, x.data, d, key, m)
        return y
    end
    ws === nothing && throw(ArgumentError("multi-factor term requires a workspace"))
    key, m = factors[1]
    dm_lmul_kernel!(ws.data, x.data, d, key, m)
    for i in 2:length(factors) - 1
        apply_kernel!(ws.data, factors[i]...)
    end
    key, m = factors[end]
    dm_axpy_kernel!(y.data, α * t.coeff, ws.data, d, key, m)
    return y
end

function _axpy!(α::Number, s::SpinOpSum, x::ST, y::ST,
                ws::Union{ST,Nothing}) where {ST<:Union{StateVector,DensityMatrix}}
    for t in s.terms
        _axpy!(α, t, x, y, ws)             # 各项 t.coeff 已计入，逐项累加
    end
    return y
end

function apply(s::SpinOpSum, s0::StateVector)
    isempty(s.terms) && throw(ArgumentError("empty SpinOpSum"))
    T = promote_type(eltype(s0), ComplexF64)
    out = StateVector(zeros(T, length(storage(s0))), _nqubits(s0))
    # 仅存在多因子项时才需要工作缓冲（纯单因子线路零额外分配）
    ws = any(t -> length(t.ops) > 1, s.terms) ?
         StateVector(similar(out.data), _nqubits(s0)) : nothing
    _axpy!(one(T), s, s0, out, ws)         # out = Σₖ tₖ·ψ
    return out
end

"""
    apply(t::SpinOpTerm, s::DensityMatrix) -> DensityMatrix
    apply(s::SpinOpSum, s0::DensityMatrix) -> DensityMatrix

把自旋算符**左乘**到密度矩阵上（`ρ ← t·ρ`，非就地，返回新密度矩阵）。
行 / 列索引独立处理：左乘只作用行索引，逐因子走局域核，
复杂度 `O(#ops · 4ⁿ)`，无需构造 `2ⁿ × 2ⁿ` 满矩阵。
"""
function apply(t::SpinOpTerm, s::DensityMatrix)
    n = _nqubits(s)
    data = copy(s.data)
    for (q, op) in reverse(t.ops)      # 逆序作用 = 左乘 A₁ A₂ ⋯
        m = _spin_matrix(op)
        1 <= q <= n || throw(ArgumentError("operator position $q out of range [1, $n]"))
        if eltype(data) <: Real && !(eltype(m) <: Real)
            data = convert(Vector{complex(float(eltype(data)))}, data)
        end
        apply_kernel!(data, (q - 1,), m)   # 只作用行索引：ρ ← m ρ
    end
    rmul!(data, t.coeff)
    return DensityMatrix(data, n)
end

function apply(s::SpinOpSum, s0::DensityMatrix)
    isempty(s.terms) && throw(ArgumentError("empty SpinOpSum"))
    T = promote_type(eltype(s0), ComplexF64)
    out = DensityMatrix(zeros(T, length(s0.data)), _nqubits(s0))
    # 仅存在多因子项时才需要工作缓冲（纯单因子线路零额外分配）
    ws = any(t -> length(t.ops) > 1, s.terms) ?
         DensityMatrix(similar(out.data), _nqubits(s0)) : nothing
    _axpy!(one(T), s, s0, out, ws)         # out = Σₖ tₖ·ρ
    return out
end

"""
    expectation(t::SpinOpTerm, s::StateVector) -> Complex
    expectation(s::SpinOpSum, s::StateVector) -> Complex
    expectation(t::SpinOpTerm, s::DensityMatrix) -> Complex
    expectation(s::SpinOpSum, s::DensityMatrix) -> Complex

自旋算符期望值：单因子项直接走 `expect_kernel` / `dm_expect_kernel`
（`⟨ψ|A|ψ⟩` 与 `tr(ρ A)`），**零分配**；多因子项仅一个工作缓冲
（`mul!` 覆盖复用）。均不构造满矩阵。
"""
function expectation(t::SpinOpTerm, s::StateVector)
    n = _nqubits(s)
    factors = _spin_factors(t, n)
    if length(factors) == 1
        key, m = factors[1]
        return t.coeff * expect_kernel(storage(s), key, m)
    end
    ws = similar(storage(s), promote_type(eltype(s), ComplexF64))
    return t.coeff * multi_expect_kernel(storage(s), factors, ws)
end

function expectation(s::SpinOpSum, st::StateVector)
    T = promote_type(eltype(st), ComplexF64)
    vw = StateVector(similar(storage(st), T), _nqubits(st))   # 工作缓冲（一次分配）
    v = storage(st)
    acc = zero(T)
    for t in s.terms
        mul!(vw, t, st)                    # vw = t·ψ，无中间态
        acc += dot(v, storage(vw))
    end
    return acc
end

function expectation(t::SpinOpTerm, s::DensityMatrix)
    n = _nqubits(s)
    d = 1 << n
    factors = _spin_factors(t, n)
    if length(factors) == 1
        key, m = factors[1]
        return t.coeff * dm_expect_kernel(s.data, d, key, m)
    end
    ws = similar(s.data, promote_type(eltype(s), ComplexF64))
    return t.coeff * dm_multi_expect_kernel(s.data, d, factors, ws)
end

function expectation(s::SpinOpSum, st::DensityMatrix)
    T = promote_type(eltype(st), ComplexF64)
    ρw = DensityMatrix(similar(st.data, T), _nqubits(st))     # 工作缓冲（一次分配）
    d = 1 << _nqubits(st)
    acc = zero(T)
    for t in s.terms
        mul!(ρw, t, st)                    # ρw = t·ρ，无中间态
        acc += sum(ρw.data[i * d + i + 1] for i in 0:d-1)   # tr(t·ρ)
    end
    return acc
end
