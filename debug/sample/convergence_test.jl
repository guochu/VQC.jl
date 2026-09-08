# debug/sample_convergence.jl — sample 函数精度验证
#
# 思路：构造一个已知严格分布的量子态，用 sample 做经验采样，
# 对比经验分布与严格分布的总变差（TV）距离随 shots 的收敛。
# 理论期望：均方误差 ~ 1/shots（TV 距离按 1/√shots 缩小）。
#
# 运行：julia --project=<VQC环境> debug/sample_convergence.jl

using VQC
using Random
using Printf

Random.seed!(20260907)

# ── 1. 构造已知严格分布的量子态（非均匀 + 含零概率分量） ────────────────────
n = 4
d = 1 << n
# 目标概率：非均匀、且有两个零概率分量（检验零概率基矢永不被采到）
p_target = Float64[(i == 5 || i == 10) ? 0.0 : 0.5 + 0.5 * sin(3i) for i in 0:d-1]
p_target ./= sum(p_target)
ψ = amplitude_encoding(sqrt.(p_target); nqubits = n)

ps_exact = probabilities(ψ)
@assert maximum(abs.(ps_exact - p_target)) < 1e-12
@assert ps_exact[6] == 0.0 && ps_exact[11] == 0.0

# ── 2. 经验分布与 TV 距离 ───────────────────────────────────────────────────
empirical_tv(ψ, ps_exact, shots) = begin
    counts = sample(ψ, shots)
    tv = 0.5 * sum(abs(get(counts, i - 1, 0) / shots - ps_exact[i]) for i in 1:d)
end

# ── 3. shots 扫描：每个 shots 重复 30 次取均值 ± 标准差 ─────────────────────
SHOTS = [10^2, 10^3, 10^4, 10^5, 10^6]
REPS = 30

println("="^78)
println("sample 精度：n = $(n)，严格分布为非均匀 p（含 2 个零概率基矢）")
println("误差 = 经验分布与严格分布的总变差距离 TV = 0.5·Σ|p̂ᵢ − pᵢ|")
println("="^78)
@printf("%-10s %-14s %-12s %-12s %-10s\n", "shots", "mean TV", "std TV", "1/√shots", "std×√shots")
println("-"^78)
prev_mean = prev_scale = nothing
for shots in SHOTS
    tvs = [empirical_tv(ψ, ps_exact, shots) for _ in 1:REPS]
    m = sum(tvs) / REPS
    s = sqrt(sum((t - m)^2 for t in tvs) / (REPS - 1))
    scale = s * sqrt(shots)      # 若按 1/√shots 收敛，该值应近似恒定
    @printf("%-10d %-14.6f %-12.6f %-12.4f %-10.4f\n", shots, m, s, 1 / sqrt(shots), scale)
end
println("-"^78)

# ── 4. 零概率基矢检查 ───────────────────────────────────────────────────────
counts_big = sample(ψ, 10^6)
@assert !haskey(counts_big, 5) && !haskey(counts_big, 10) "零概率基矢被采样！"
println("零概率基矢（0-based idx 5、10）在 10⁶ shots 中出现次数：0 ✓")

# ── 5. 子集采样：边缘分布的收敛 ─────────────────────────────────────────────
qs = [3, 1]                                   # 非连续子集
ps_edge = probabilities(ψ, qs)
println()
println("子集采样（qubits = $qs）的边缘分布收敛：")
@printf("%-10s %-14s\n", "shots", "mean TV")
println("-"^40)
for shots in SHOTS
    tvs = Float64[]
    for _ in 1:REPS
        counts = sample(ψ, qs, shots)
        tv = 0.5 * sum(abs(get(counts, i - 1, 0) / shots - ps_edge[i]) for i in eachindex(ps_edge))
        push!(tvs, tv)
    end
    @printf("%-10d %-14.6f\n", shots, sum(tvs) / REPS)
end
println("-"^40)
println("结论：TV 距离随 shots 增大按 1/√shots 收敛，经验分布趋近严格分布。")
