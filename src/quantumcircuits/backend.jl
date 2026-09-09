# backend.jl — QuantumCircuits.Interface 的后端实现
#
# `StateVectorBackend`：把 `Circuit` 演化到 `StateVector` 上的
# QuantumCircuits.Interface 后端。支持中途测量 / 经典反馈（IfOp）、
# 信道指令；`shots > 0` 时逐 shot 采样，按 `counts_key` 约定聚合计数。

using QuantumCircuits.Interface: Backend, SimResult
import QuantumCircuits.Interface: simulate

"""
    StateVectorBackend(; dtype = ComplexF64) <: QuantumCircuits.Interface.Backend

态矢量后端：`QuantumCircuits.Interface.simulate(c, backend; shots, seed)`
把整条线路演化到 `StateVector` 上。

* `shots = 0`：强模拟，`SimResult.state` 给出终态；
* `shots > 0`：逐 shot 采样，测量 / 经典反馈写回经典寄存器，
  `SimResult.counts` 按 `counts_key` 约定聚合；
* 能力：`:statevector`、`:mid_measure`、`:noise`。
"""
struct StateVectorBackend <: Backend
    dtype::DataType
end

StateVectorBackend(; dtype::DataType = ComplexF64) = StateVectorBackend(dtype)

QuantumCircuits.Interface.supports(::StateVectorBackend, cap::Symbol) =
    cap in (:statevector, :mid_measure, :noise)

# 线路中含 MeasOp 的经典寄存器（计数键只覆盖真正被测量的寄存器）
function _measured_cregs(c::Circuit)
    out = CReg[]
    for op in c.ops
        op isa MeasOp || continue
        for cb in op.clbits
            cb.reg in out || push!(out, cb.reg)
        end
    end
    return out
end

function QuantumCircuits.Interface.simulate(c::Circuit, b::StateVectorBackend;
                                            shots::Int = 0,
                                            seed::Union{Nothing,Integer} = nothing,
                                            kwargs...)
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    measured = _measured_cregs(c)
    if shots <= 0
        store = ClassicalStore(c)
        state = _simulate_bound!(c, zero_state(b.dtype, c.n), _ParamEnv(), store)
        return SimResult(nothing, state)
    end

    counts = Dict{String,Int}()
    for _ in 1:shots
        store = ClassicalStore(c)
        s = _simulate_bound!(c, zero_state(b.dtype, c.n), _ParamEnv(), store)
        outcome = Dict{ClbitRef,Int}()
        for reg in measured
            v = _reg_value(store, reg)
            for i in 1:reg.n
                outcome[ClbitRef(reg, i)] = (v >> (i - 1)) & 1
            end
        end
        k = counts_key(c, outcome)
        counts[k] = get(counts, k, 0) + 1
    end
    return SimResult(; counts = counts,
                     metadata = Dict{Symbol,Any}(:shots => shots))
end
