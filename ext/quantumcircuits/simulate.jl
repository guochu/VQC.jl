# simulate.jl — 线路演化

# ── 参数环境 ─────────────────────────────────────────────────────────────────

"""
    _ParamEnv <: AbstractDict{Param,Float64}

模拟期的参数环境：按 `parameters(c)` 顺序存放绑定值 `vec`，并提供
`Param → 序号` 的 `index`。继承 `AbstractDict` 使其可直接供
`QuantumCircuits.mat(op, table)` 查表；`vec` 的向量索引对自动微分友好。
"""
struct _ParamEnv <: AbstractDict{Param,Float64}
    vec::Vector{Float64}
    index::Dict{Param,Int}

    function _ParamEnv(vec::Vector{Float64}, index::Dict{Param,Int})
        length(vec) == length(index) ||
            throw(ArgumentError("parameter values and index length mismatch"))
        new(vec, index)
    end
end

_ParamEnv() = _ParamEnv(Float64[], Dict{Param,Int}())

Base.getindex(e::_ParamEnv, p::Param) = e.vec[e.index[p]]
Base.haskey(e::_ParamEnv, p::Param) = haskey(e.index, p)
Base.keys(e::_ParamEnv) = keys(e.index)
Base.length(e::_ParamEnv) = length(e.vec)
Base.iterate(e::_ParamEnv, state...) = iterate(pairs(e.index), state...)

"""
    _param_env(c, params) -> _ParamEnv

* `params::Vector{<:Real}`：按 `parameters(c)` 顺序绑定；
* `params::Dict`：参数表（键可为 `Param` / `Symbol` / `ParamVector`）；
* `nothing`：不绑定。
"""
function _param_env(c::Circuit, params::AbstractVector{<:Real})
    ps = parameters(c)
    length(params) == length(ps) ||
        throw(ArgumentError("circuit has $(length(ps)) symbolic parameter(s), got $(length(params)) values"))
    return _ParamEnv(Float64.(params), Dict{Param,Int}(p => i for (i, p) in enumerate(ps)))
end

function _param_env(c::Circuit, table::AbstractDict)
    ps = parameters(c)
    t = Dict{Param,Float64}()
    for (k, v) in table
        if k isa ParamVector
            for (i, vv) in enumerate(v)
                t[k[i]] = Float64(vv)
            end
        elseif k isa Param
            t[k] = Float64(v)
        elseif k isa Symbol
            t[Param(k)] = Float64(v)
        else
            throw(ArgumentError("invalid parameter key $(k)"))
        end
    end
    vec = Float64[get(t, p) do
                      throw(ArgumentError("parameter $(p.name) is not bound"))
                  end for p in ps]
    return _ParamEnv(vec, Dict{Param,Int}(p => i for (i, p) in enumerate(ps)))
end

_param_env(::Circuit, ::Nothing) = _ParamEnv()

# ── 演化 ─────────────────────────────────────────────────────────────────────

"""
    simulate(c::Circuit, state; params=nothing) -> state'
    simulate!(c::Circuit, state; params=nothing) -> state
    c * state

演化整条线路（`simulate` 非就地；`simulate!` 就地）。测量结果写回
经典寄存器，供 `IfOp` 条件分支使用。

`params`：

* `Vector{<:Real}`：按 `parameters(c)` 顺序绑定符号参数；
* `Dict`：参数表（键可为 `Param` / `Symbol` / `ParamVector`）；
* `nothing`：不绑定（线路须无未绑定符号参数）。

自动微分提示：`params` 传 **向量** 时整条链路可微（见包扩展
`VQCZygoteExt`）；传 `Dict` 时参数梯度不可用。
"""
function simulate(c::Circuit, s::Union{StateVector,DensityMatrix}; params=nothing)
    return _simulate_bound(c, copy(s), _param_env(c, params))
end

function simulate!(c::Circuit, s::Union{StateVector,DensityMatrix}; params=nothing)
    return _simulate_bound!(c, s, _param_env(c, params))
end

function _simulate_bound(c::Circuit, s::Union{StateVector,DensityMatrix}, env::_ParamEnv)
    return _simulate_bound!(c, copy(s), env)
end

function _simulate_bound!(c::Circuit, s::Union{StateVector,DensityMatrix}, env::_ParamEnv)
    store = ClassicalStore(c)
    for op in c.ops
        s = _apply_with_store!(s, op, store, env)
    end
    return s
end

function _apply_with_store!(s, op::Operation, store::ClassicalStore, env::_ParamEnv)
    return apply!(s, op, env)
end

function _apply_with_store!(s, op::MeasOp, store::ClassicalStore, env::_ParamEnv)
    for (q, cb) in zip(op.qubits, op.clbits)
        outcome, _ = measure!(s, q)
        _set_clbit!(store, cb, outcome)
    end
    return s
end

function _apply_with_store!(s, op::IfOp, store::ClassicalStore, env::_ParamEnv)
    return apply!(s, op, store, env)
end

# ── 便利运算符 ───────────────────────────────────────────────────────────────

Base.:*(c::Circuit, s::Union{StateVector,DensityMatrix}) = simulate(c, s)
Base.:*(op::GateOp, s::StateVector) = apply!(copy(s), op)
Base.:*(op::GateOp, s::DensityMatrix) = apply!(copy(s), op)
