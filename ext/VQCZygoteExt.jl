# =============================================================================
# VQCZygoteExt — VQC 的自动微分扩展（`using Zygote` 时自动加载）
#
# 能力：
#   * `simulate(c, ψ; params=θ)`：线路整体对初始态与参数向量求导；
#   * 一般参数门的梯度：对 `ParamGate.matrix_fn` 走 Zygote 精确微分，
#     不可微时自动回退到中心差分（因此**不限于 Rx/Ry/Rz**，任意
#     `ParamGate` / `UserGate` / 修饰门 `inv/pow/ctrl` 均可求导）；
#   * 噪声量子线路的梯度：`ChannelOp` 作用在 `DensityMatrix` 上，
#     通过信道超算符的伴随映射回传；
#   * `expectation(PauliSum/PauliTerm, ψ)` 与 `post_select` 的伴随；
#   * 符号参数权重共享：同名 `Param` 的梯度自动累加。
#
# 约束：
#   * 含噪线路的求导须全程使用 `DensityMatrix`；
#   * `MeasOp` / `ReinitOp` / `IfOp` 不可微（反向经过时抛错）；
#     `BarrierOp` / 全酉 `BlockOp` 透明回传。
# =============================================================================

module VQCZygoteExt

using Zygote
using Zygote: @adjoint
using LinearAlgebra
using QuantumCircuits
using QuantumCircuits: Gate, ConstGate, ParamGate, UserGate, InvGate, PowGate, CtrlGate,
                       GateOp, ChannelOp, BarrierOp, BlockOp, MeasOp, ReinitOp, IfOp,
                       Operation, Circuit, Channel, Param,
                       qubits, mat, kraus, nqubits, parameters, unroll
using QuantumCircuits: _embed, _controlled_matrix
using VQC
using VQC: apply_kernel!, _lsb_key, _sorted_key, _offsets, _bit_insert_zeros, post_select,
           expect_kernel
# 本扩展的触发集（Zygote + QuantumCircuits）包含接口扩展的触发集
# （QuantumCircuits），因此接口扩展必已加载，可安全取用其内部机制。
const _QCI = Base.get_extension(VQC, :VQCQuantumCircuitsExt)
const _pauli_local_matrix = _QCI._pauli_local_matrix
const _param_env = _QCI._param_env
const _simulate_bound = _QCI._simulate_bound
const _ParamEnv = _QCI._ParamEnv
# 被 @adjoint 扩展的函数必须以 import 方式引入
import VQC: StateVector, DensityMatrix, storage

# ── 基础伴随 ──────────────────────────────────────────────────────────────────

@adjoint storage(x::Union{StateVector,DensityMatrix}) = storage(x),
    z -> begin
        # getfield 语义：z 是 NamedTuple(data=..., n=...)
        dz = z isa NamedTuple ? get(z, :data, nothing) : z
        (typeof(x)(dz, VQC._nqubits(x)),)
    end

@adjoint nqubits(x::Union{StateVector,DensityMatrix}) = nqubits(x), _ -> (nothing,)

@adjoint function StateVector(data::AbstractVector{<:Number}, n::Int)
    StateVector(data, n), z -> (storage(z), nothing)
end

@adjoint function StateVector(data::AbstractVector{<:Number})
    StateVector(data), z -> (storage(z),)
end

@adjoint function DensityMatrix(data::AbstractVector{<:Number}, n::Int)
    DensityMatrix(data, n), z -> (storage(z), nothing)
end

@adjoint function DensityMatrix(data::AbstractMatrix{<:Number}, n::Int)
    DensityMatrix(data, n), z -> (storage(z), nothing)
end

@adjoint function DensityMatrix(data::AbstractMatrix{<:Number})
    DensityMatrix(data), z -> (storage(z),)
end

# 纯态 → 密度矩阵：ρ = ψψ†，态余切 G = Δψ + Δ†ψ。
# 余切经 kwargs 动态边界可能变为结构体形式 NamedTuple{data, n}，须解包。
@adjoint function DensityMatrix(x::StateVector)
    ρ = DensityMatrix(x)
    return ρ, Δ -> begin
        Δm = Δ isa NamedTuple ? Δ.data :
             (Δ isa DensityMatrix ? storage(Δ) : Δ)
        ps = storage(x)
        (StateVector(Δm * ps + adjoint(Δm) * ps, VQC._nqubits(x)),)
    end
end

# 无输入依赖的态构造：梯度为空
@adjoint function VQC.zero_state(args...)
    VQC.zero_state(args...), _ -> nothing
end

@adjoint function VQC.rand_state(args...)
    VQC.rand_state(args...), _ -> nothing
end

# ── 态上直接作用矩阵的内部工具 ────────────────────────────────────────────────

function _apply_sv_matrix!(s::StateVector, qs::Vector{Int}, m::AbstractMatrix)
    apply_kernel!(storage(s), _lsb_key(qs), m)
    return s
end

function _apply_dm_matrix!(s::DensityMatrix, qs::Vector{Int}, m::AbstractMatrix)
    n = nqubits(s)
    key = _lsb_key(qs)
    apply_kernel!(s.data, key, m)
    apply_kernel!(s.data, ntuple(i -> key[i] + n, Val(length(qs))), conj(m))
    return s
end

_apply_back!(Δ::StateVector, qs::Vector{Int}, m::AbstractMatrix) = _apply_sv_matrix!(Δ, qs, m)
_apply_back!(Δ::DensityMatrix, qs::Vector{Int}, m::AbstractMatrix) = _apply_dm_matrix!(Δ, qs, m)

# ── 一般参数门的雅可比矩阵 ────────────────────────────────────────────────────

"""
    _gate_jacobians(g::Gate, θ::Vector{Float64}) -> Vector{Matrix{ComplexF64}}

返回 `[∂U/∂θ₁, …, ∂U/∂θₙ]`。优先对 `matrix_fn` 做 Zygote 精确微分，
失败则回退到中心差分；修饰门（`ctrl`/`inv`）走结构化规则，`pow` 与
复合 `UserGate` 走差分。
"""
function _gate_jacobians(g::Gate, θ::Vector{Float64})
    return _fd_jacobians(g, θ)
end

function _gate_jacobians(g::ParamGate, θ::Vector{Float64})
    n = 1 << nqubits(g)
    try
        J = first(Zygote.jacobian(p -> vec(g.matrix_fn(p...)), θ))
        return [reshape(ComplexF64.(view(J, :, j)), n, n) for j in eachindex(θ)]
    catch
        return _fd_jacobians(g, θ)
    end
end

"逆门的导数：∂U†/∂θ = (∂U/∂θ)†（精确）。"
function _gate_jacobians(g::InvGate, θ::Vector{Float64})
    return [adjoint(dU) for dU in _gate_jacobians(g.g, θ)]
end

"受控门的导数：∂(ctrl U)/∂θ = ctrl(∂U/∂θ)（精确，线性）。"
function _gate_jacobians(g::CtrlGate, θ::Vector{Float64})
    return [_controlled_matrix(dU, g.negs) for dU in _gate_jacobians(g.g, θ)]
end

function _gate_jacobians(g::Union{PowGate,UserGate}, θ::Vector{Float64})
    return _fd_jacobians(g, θ)
end

"中心差分回退（相对精度 ~1e-9）。"
function _fd_jacobians(g::Gate, θ::Vector{Float64})
    out = Vector{Matrix{ComplexF64}}(undef, length(θ))
    for j in eachindex(θ)
        h = 1e-6 * max(1.0, abs(θ[j]))
        θp = copy(θ); θp[j] += h
        θm = copy(θ); θm[j] -= h
        out[j] = (Matrix{ComplexF64}(mat(g, θp)) - Matrix{ComplexF64}(mat(g, θm))) ./ (2h)
    end
    return out
end

"门参数的绑定值（符号参数查表，常量直取）。"
function _bound_params(op::GateOp, table::Union{Nothing,AbstractDict})
    return Float64[p isa Param ? _lookup_param(p, table) : p for p in op.params]
end

function _lookup_param(p::Param, table::Union{Nothing,AbstractDict})
    table === nothing && throw(ArgumentError("unbound parameter $(p.name)"))
    haskey(table, p) && return Float64(table[p])
    haskey(table, p.name) && return Float64(table[p.name])
    throw(ArgumentError("unbound parameter $(p.name)"))
end

# ── 门反传：拉回 Δ 与 y，并累加参数梯度 ──────────────────────────────────────

# 约定：态余切 Δ 满足 dL = Re⟨Δ|dψ⟩（SV）/ dL = Re tr(Δ†dρ)（DM）。
# SV 梯度公式：dL/dθ = Re⟨Δ_out|∂U|y_in⟩（Δ 不拉回、y 取正向记录的门输入态）。
# DM 梯度公式（全局矩阵形式，U 为门矩阵、dU = ∂U）：
#   dL/dθ = Re tr(dU·ρ_in U†Δ†) + Re tr(dU·conj(Δ†U ρ_in))。
_grad_contrib(y::StateVector, U, dU, Δ::StateVector, qs::Vector{Int}) =
    real(expect_kernel(storage(Δ), storage(y), _lsb_key(qs), dU))

function _grad_contrib(y::DensityMatrix, U, dU, Δ::DensityMatrix, qs::Vector{Int})
    n = nqubits(y)
    Ug = _embed(Matrix{ComplexF64}(U), qs, n)
    dUg = _embed(Matrix{ComplexF64}(dU), qs, n)
    ρs = storage(y)
    Δs = storage(Δ)
    g1 = real(sum(dUg .* transpose(ρs * adjoint(Ug) * adjoint(Δs))))
    g2 = real(sum(conj(dUg) .* transpose(adjoint(Δs) * Ug * ρs)))
    return g1 + g2
end

function _back_gateop!(Δ::Union{StateVector,DensityMatrix}, y::Union{StateVector,DensityMatrix},
                       op::GateOp, table::Union{Nothing,AbstractDict}, acc::Dict{Param,Float64})
    qs = qubits(op)
    m = mat(op, table)
    # y 为正向记录的门输入态（非酉参数门下无法用 U† 拉回，须真实中间态）
    if any(p -> p isa Param, op.params)
        θ = _bound_params(op, table)
        dUs = _gate_jacobians(op.gate, θ)
        for (j, dU) in enumerate(dUs)
            p = op.params[j]
            p isa Param || continue
            g = _grad_contrib(y, m, dU, Δ, qs)
            acc[p] = get(acc, p, 0.0) + g
        end
    end
    _apply_back!(Δ, qs, adjoint(m))
    return
end

# ── 噪声信道反传（仅 DensityMatrix） ─────────────────────────────────────────

"""
    _supermat(ch::Channel) -> Matrix

信道超算符 `S = Σₖ kron(conj(Kₖ), Kₖ)`：向量化约定
`vec(ρ)[r + d*c]`，局域索引 `a = r_loc + d*c_loc`（行占低位）。
"""
function _supermat(ch::Channel)
    ks = kraus(ch)
    d = size(ks[1], 1)
    S = zeros(ComplexF64, d * d, d * d)
    for K in ks
        S .+= kron(conj(K), K)
    end
    return S
end

function _back_channelop!(Δ::DensityMatrix, op::ChannelOp)
    n = nqubits(Δ)
    qs = qubits(op)
    all(q -> 1 <= q <= n, qs) || throw(ArgumentError("channel qubit out of range"))
    Nq = length(qs)
    key = _lsb_key(qs)
    key2 = ntuple(i -> (i <= Nq ? key[i] : key[i-Nq] + n), Val(2Nq))
    # Δ ← Σₖ Kₖ† Δ Kₖ（伴随映射：S† 分块作用于 vec(ρ)）
    apply_kernel!(Δ.data, key2, adjoint(_supermat(op.channel)))
    return
end

# ── 反向遍历（正向记录中间态 + 反向回传） ─────────────────────────────────────

"展开 BlockOp 得到平铺指令列表。"
function _flatten_ops!(out::Vector{Operation}, ops)
    for op in ops
        if op isa BlockOp
            _flatten_ops!(out, unroll(op))
        else
            push!(out, op)
        end
    end
    return out
end
_flatten_ops(ops) = _flatten_ops!(Operation[], ops)

"""
正向重放：记录每条平铺指令的输入态（门梯度需要真实中间态——
非酉参数门无法用 U† 拉回）。`MeasOp` / `ReinitOp` / `IfOp` 不可微，
重放即抛 `ArgumentError`。
"""
function _forward_states(ops::Vector{Operation}, s::Union{StateVector,DensityMatrix},
                         env::_ParamEnv)
    states = Vector{Union{StateVector,DensityMatrix}}(undef, length(ops))
    cur = copy(s)
    for (k, op) in enumerate(ops)
        states[k] = copy(cur)   # 必须拷贝：apply! 就地修改 cur
        if op isa GateOp || op isa ChannelOp || op isa BarrierOp
            cur = apply!(cur, op, env)
        else
            throw(ArgumentError("VQC AD cannot differentiate through $(op) ($(typeof(op))); " *
                                "supported: GateOp / ChannelOp / BarrierOp / BlockOp"))
        end
    end
    return states
end

function _back_walk!(Δ::Union{StateVector,DensityMatrix},
                     states::Vector{<:Union{StateVector,DensityMatrix}},
                     ops::Vector{Operation}, table::Union{Nothing,AbstractDict},
                     acc::Dict{Param,Float64})
    for k in length(ops):-1:1
        op = ops[k]
        if op isa GateOp
            _back_gateop!(Δ, states[k], op, table, acc)
        elseif op isa ChannelOp
            Δ isa DensityMatrix ||
                throw(ArgumentError("noisy-circuit AD requires DensityMatrix throughout; " *
                                    "got cotangent of type $(typeof(Δ)) at channel $(op)"))
            _back_channelop!(Δ, op)
        elseif op isa BarrierOp
            # 无语义，透明回传
        else
            throw(ArgumentError("VQC AD cannot differentiate through $(op) ($(typeof(op))); " *
                                "supported: GateOp / ChannelOp / BarrierOp / BlockOp"))
        end
    end
    return Δ
end

# ── simulate / simulate! 的伴随 ──────────────────────────────────────────────

function _simulate_pullback(c::Circuit, s::Union{StateVector,DensityMatrix},
                            out::Union{StateVector,DensityMatrix},
                            env::_ParamEnv, Δ)
    Δ2 = copy(Δ)
    if Δ2 isa DensityMatrix && s isa StateVector
        throw(ArgumentError("noisy-circuit AD requires a DensityMatrix input state " *
                            "(got StateVector input but DensityMatrix cotangent)"))
    end
    ops = _flatten_ops(c.ops)
    states = _forward_states(ops, s, env)
    acc = Dict{Param,Float64}()
    _back_walk!(Δ2, states, ops, env, acc)
    gvec = Float64[get(acc, p, 0.0) for p in parameters(c)]
    return (nothing, Δ2, (vec = gvec, index = nothing))
end

for S in (:StateVector, :DensityMatrix)
    @eval begin
        @adjoint function _simulate_bound(
            c::Circuit, s::$S, env::_ParamEnv,
        )
            out = _simulate_bound(c, s, env)
            return out, Δ -> _simulate_pullback(c, s, out, env, Δ)
        end
    end
end

# ── 参数环境的伴随：梯度经 env.vec 流回 params 向量 ─────────────────────────

@adjoint function _ParamEnv(vec::Vector{Float64}, index::Dict{Param,Int})
    env = _ParamEnv(vec, index)
    return env, denv -> begin
        dvec = denv isa NamedTuple ? get(denv, :vec, nothing) : nothing
        ((dvec, nothing),)
    end
end

@adjoint function _param_env(c::Circuit, params::AbstractVector{<:Real})
    env = _param_env(c, params)
    return env, denv -> begin
        dvec = denv isa NamedTuple ? get(denv, :vec, nothing) : nothing
        (nothing, dvec)
    end
end

@adjoint function _param_env(c::Circuit, ::Nothing)
    _param_env(c, nothing), _ -> nothing
end

@adjoint function _param_env(c::Circuit, table::AbstractDict)
    _param_env(c, table), _ -> nothing
end

# ── expectation 的伴随 ────────────────────────────────────────────────────────

"""
    _expect_sv_adj(z, s, qs, M)

纯态可观测量伴随：`E = ⟨ψ|M|ψ⟩` 的态余切
`grad = conj(z)·Mψ + z·M†ψ`。
"""
function _expect_sv_adj(z, s::StateVector, qs::Vector{Int}, M::AbstractMatrix)
    g = zero(storage(s))
    tmp = similar(g)
    copyto!(tmp, storage(s))
    apply_kernel!(tmp, _lsb_key(qs), M)
    @. g += conj(z) * tmp
    copyto!(tmp, storage(s))
    apply_kernel!(tmp, _lsb_key(qs), adjoint(M))
    @. g += z * tmp
    return StateVector(g, nqubits(s))
end

"""
    _expect_dm_adj(z, s, qs, M)

混合态可观测量伴随：`E = tr(ρ M)` 的态余切 `z·M†`（全矩阵嵌入）。
"""
function _expect_dm_adj(z, s::DensityMatrix, qs::Vector{Int}, M::AbstractMatrix)
    n = nqubits(s)
    M_full = _embed(Matrix{ComplexF64}(M), reverse(qs), n)
    return DensityMatrix(z .* adjoint(M_full), n)
end

_pauli_qs(t::QuantumCircuits.Hamiltonian.PauliTerm) = Int[q for (q, _) in t.ops]

function _pauli_full_local(t::QuantumCircuits.Hamiltonian.PauliTerm)
    isempty(t.ops) && throw(ArgumentError("identity term has no local matrix"))
    return t.coeff .* _pauli_local_matrix(t.ops)
end

for S in (:StateVector, :DensityMatrix)
    @eval begin
        @adjoint function VQC.expectation(t::QuantumCircuits.Hamiltonian.PauliTerm, s::$S)
            val = VQC.expectation(t, s)
            return val, z -> begin
                isempty(t.ops) && return (nothing, zero(s))
                qs = _pauli_qs(t)
                M = _pauli_full_local(t)
                grad = ($S === StateVector ? _expect_sv_adj(z, s, qs, M) : _expect_dm_adj(z, s, qs, M))
                return (nothing, grad)
            end
        end

        @adjoint function VQC.expectation(h::QuantumCircuits.Hamiltonian.PauliSum, s::$S)
            val = VQC.expectation(h, s)
            return val, z -> begin
                grad = zero(s)
                for t in h.terms
                    isempty(t.ops) && continue
                    qs = _pauli_qs(t)
                    M = _pauli_full_local(t)
                    g = ($S === StateVector ? _expect_sv_adj(z, s, qs, M) : _expect_dm_adj(z, s, qs, M))
                    grad += g
                end
                return (nothing, grad)
            end
        end
    end
end

# ── post_select 的伴随 ────────────────────────────────────────────────────────

"投影算子 `P(q,val)` 作用到态（非匹配分支清零，不归一化；q 为 1-based）。"
function _project_apply(s::StateVector, q::Int, val::Int)
    q -= 1
    v = copy(storage(s))
    @inbounds for i in 0:length(v)-1
        ((i >> q) & 1) != val && (v[i+1] = zero(eltype(v)))
    end
    return StateVector(v, nqubits(s))
end

function _project_apply(s::DensityMatrix, q::Int, val::Int)
    q -= 1
    d = 1 << nqubits(s)
    m = copy(storage(s))
    @inbounds for i in 0:d-1, j in 0:d-1
        (((i >> q) & 1) != val || ((j >> q) & 1) != val) && (m[i+1, j+1] = zero(eltype(m)))
    end
    return DensityMatrix(vec(m), nqubits(s))
end

for S in (:StateVector, :DensityMatrix)
    @eval begin
        # 前向：s' = P s/√p，p = ⟨s|P|s⟩（SV）/ tr(Pρ)（DM），比特数不变。
        # 伴随：grad_s = P·Δs'/√p + (2Δp − Re⟨Δs'|s'⟩/p)·Ps   （SV）
        #       grad_ρ = P·Δρ'·P/p + (Δp − Re tr(Δ†ρ')/p)·P    （DM）
        @adjoint function VQC.post_select(s::$S, q::Integer, val::Integer)
            out = VQC.post_select(s, q, val)
            st, p = out
            return out, Δ -> begin
                Δst = Δ === nothing ? nothing : Δ[1]
                Δp = Δ === nothing ? nothing : Δ[2]
                (Δst === nothing && Δp === nothing) && return (nothing, nothing)
                Ps = _project_apply(s, Int(q), Int(val))
                if ($S === StateVector)
                    coef = (Δp === nothing ? 0.0 : 2 * Δp)
                    if Δst !== nothing
                        coef -= real(dot(storage(Δst), storage(st))) / p
                    end
                    g = Δst === nothing ? zero(storage(Ps)) : storage(_project_apply(Δst, Int(q), Int(val))) ./ sqrt(p)
                    g = coef == 0 ? g : g .+ coef .* storage(Ps)
                    return (StateVector(g, nqubits(s)), nothing)
                else
                    coef = (Δp === nothing ? 0.0 : Δp)
                    if Δst !== nothing
                        coef -= real(tr(storage(st)' * storage(Δst))) / p
                    end
                    g = Δst === nothing ? zero(storage(Ps)) : storage(_project_apply(Δst, Int(q), Int(val))) ./ p
                    g = coef == 0 ? g : g .+ coef .* storage(Ps)
                    return (DensityMatrix(vec(g), nqubits(s)), nothing)
                end
            end
        end
    end
end

end # module
