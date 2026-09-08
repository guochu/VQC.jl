# accuracy.jl — 精度测试：VQC vs Yao，1..20 比特随机线路
#
# 用法：julia --project=... --threads=<N> accuracy.jl <tag>
#   （tag 用于区分线程配置，如 "1" / "auto"）
#
# 对每个 n ∈ 1..20、每种 BlasFloat、每种 kernel batch size
# （8 / 16 / 32 / 64）：构造随机线路，VQC 与 Yao 各自独立作用同一
# 初态，比较终态相对误差（应 ≤ 1e-5）。

include("common.jl")

tag = length(ARGS) >= 1 ? ARGS[1] : string(Threads.nthreads())
rng = MersenneTwister(20260906)
dir = ensure_results_dir()
path = joinpath(dir, "accuracy_t$tag.csv")
BATCHSIZES = (8, 16, 32, 64)

println("="^72)
println("accuracy: VQC vs Yao（线程 = ", Threads.nthreads(),
        "，batch size = ", BATCHSIZES, "，结果写入 $path）")
println("="^72)

open(path, "w") do io
    println(io, "n,eltype,bs,threads,err")
    for K in BATCHSIZES
        VQC.set_kernel_batch!(K)
        for n in 1:20
            for T in TYPES
                gates = random_circuit(rng, n, T)
                v0 = randn(rng, T, 1 << n)
                v_vqc = run_vqc(copy(v0), gates, n)
                v_yao = run_yao(copy(v0), gates, n)
                err = norm(v_vqc - v_yao) / norm(v_yao)
                ok = err < 1e-5
                println(io, "$n,$T,$K,$(Threads.nthreads()),$err")
                @printf("bs=%2d  %3d  %-10s  err = %.3e  %s\n",
                        K, n, T, err, ok ? "OK" : "FAIL <<<")
            end
        end
    end
end
println("完成。")
