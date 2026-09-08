# common.jl — accuracy.jl / perf.jl 共用的工具
#
# 约定：
#   * 精度参考：VQC 与 Yao 各自独立跑同一条随机线路，比较终态
#     （两者均与稠密矩阵语义对齐，互相对拍）；
#   * 线程数由进程启动参数 `--threads` 决定，结果 CSV 中记录
#     `Threads.nthreads()`；
#   * 4 种 BlasFloat：Float32 / Float64 / ComplexF32 / ComplexF64。

using Random
using LinearAlgebra
using Printf
using VQC
using Yao

const TYPES = (Float32, Float64, ComplexF32, ComplexF64)

"随机酉 / 正交矩阵（保持 eltype 语义：实数类型生成正交阵）。"
rand_unitary(::Type{T}, D::Int) where {T<:AbstractFloat} = Matrix{T}(qr(randn(T, D, D)).Q)
rand_unitary(::Type{Complex{T}}, D::Int) where {T<:AbstractFloat} =
    Matrix{Complex{T}}(qr(randn(Complex{T}, D, D)).Q)

"把 MSB-first 行序的矩阵重排为 LSB-first 行序（Yao 的 matblock 约定）。"
function msb_to_lsb(U::AbstractMatrix, nb::Int)
    D = 1 << nb
    out = similar(U)
    rev(r) = sum(((r >> (nb - 1 - i)) & 1) << i for i in 0:nb-1)
    for r in 0:D-1, c in 0:D-1
        out[rev(r)+1, rev(c)+1] = U[r+1, c+1]
    end
    return out
end

"随机线路：`depth` 层随机单 / 两比特门（两比特为连续对）。"
function random_circuit(rng::AbstractRNG, n::Int, T::Type; depth::Int = clamp(2n, 4, 40))
    gates = Vector{Tuple{Matrix,Vector{Int}}}()
    for _ in 1:depth
        if n == 1 || rand(rng, Bool)
            push!(gates, (rand_unitary(T, 2), [rand(rng, 1:n)]))
        else
            k = rand(rng, 1:n-1)
            push!(gates, (rand_unitary(T, 4), [k, k + 1]))
        end
    end
    return gates
end

"用 VQC 跑线路（就地，返回终态向量）。"
function run_vqc(v0::Vector{T}, gates, n::Int) where {T}
    sv = StateVector(v0, n)
    for (U, locs) in gates
        apply!(sv, U, locs)
    end
    return v0
end

"用 Yao 跑线路（就地，返回终态向量）。"
function run_yao(v0::Vector{T}, gates, n::Int) where {T}
    reg = ArrayReg(v0)
    for (U, locs) in gates
        Yao.apply!(reg, put(n, (sort(locs)...,) => matblock(msb_to_lsb(U, length(locs)))))
    end
    return vec(Yao.statevec(reg))
end

"耗时（秒）：预热 1 次 + 计时取 `n` 次最小值。"
function bench_min(f; n::Int = 3, warmup::Bool = true)
    warmup && f()
    best = Inf
    for _ in 1:n
        t = @elapsed f()
        best = min(best, t)
    end
    return best
end

function ensure_results_dir()
    d = joinpath(@__DIR__, "results")
    isdir(d) || mkpath(d)
    return d
end
