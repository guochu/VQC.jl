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

"把 MSB-first 的 U（作用在 locs 上，locs[1] 为最高位）重排为 Yao 的
升序 locs 位序矩阵：Yao 的 matblock 总是先把 locs 排序后按自然位权作用，
任意（含乱序 / 非连续）位置的矩阵必须这样换轴才能与 VQC 语义一致。"
function permute_matrix(U::AbstractMatrix, locs::NTuple{N,Int}) where {N}
    D = 1 << N
    s = Tuple(sort(collect(locs)))                     # Yao 内部排序后的位序
    j_of = Dict(q => j for (j, q) in enumerate(locs))  # VQC 位权 = N - j（j=1 为最高位）
    perm(a::Int) = begin                               # Yao 位序行索引 → VQC 位权行索引
        r = 0
        for i in 0:N-1
            q = s[i+1]
            r |= ((a >> i) & 1) << (N - j_of[q])
        end
        return r
    end
    out = Matrix{eltype(U)}(undef, D, D)
    for a in 0:D-1, b in 0:D-1
        out[a+1, b+1] = U[perm(a)+1, perm(b)+1]
    end
    return out
end

"均匀分散在 1..n 的 w 个位置（spread 模式；w==1 取中间位）。"
function spread_locs(n::Int, w::Int)
    w == 1 && return (cld(n, 2),)
    return Tuple(sort(unique(round.(Int, range(1, n; length = w)))))
end

"随机线路：宽度 1..min(5,n) 的随机酉门，位置取“随机子集 / 连续块 /
均匀分散”多种模式，另补足 `depth` 个随机门（随机宽度 + 随机子集位置）。"
function random_circuit(rng::AbstractRNG, n::Int, T::Type; depth::Int = clamp(2n, 8, 40))
    maxw = min(5, n)
    gates = Vector{Tuple{Matrix{T},Tuple{Vararg{Int}}}}()
    # —— 每种门宽的系统化位置覆盖 ——
    for w in 1:maxw
        D = 1 << w
        p = randperm(rng, n)                           # 随机子集（顺序任意，覆盖降序/非线性位）
        push!(gates, (rand_unitary(T, D), Tuple(p[1:w])))
        w == 1 && continue
        off = rand(rng, 1:n-w+1)                       # 连续块（随机偏移）
        push!(gates, (rand_unitary(T, D), Tuple(off:off+w-1)))
        if n >= 2w - 1 && length(spread_locs(n, w)) == w   # 均匀分散
            push!(gates, (rand_unitary(T, D), spread_locs(n, w)))
        end
    end
    # —— 补足深度：随机宽度 + 随机位置 ——
    while length(gates) < depth
        w = rand(rng, 1:maxw)
        p = randperm(rng, n)
        push!(gates, (rand_unitary(T, 1 << w), Tuple(p[1:w])))
    end
    return gates
end

"用 VQC 跑线路（就地，返回终态向量）。"
function run_vqc(v0::Vector{T}, gates, n::Int) where {T}
    sv = StateVector(v0, n)
    for (U, locs) in gates
        VQC.apply!(sv, U, locs)
    end
    return v0
end

"用 Yao 跑线路（就地，返回终态向量）。任意位置的门先做轴重排
（permute_matrix），再按 Yao 的升序 locs 语义施加。"
function run_yao(v0::Vector{T}, gates, n::Int) where {T}
    reg = ArrayReg(v0)
    for (U, locs) in gates
        s = Tuple(sort(collect(locs)))
        Yao.apply!(reg, put(n, s => matblock(permute_matrix(U, locs))))
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
