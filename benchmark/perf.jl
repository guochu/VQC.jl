# perf.jl — 性能测试：n ∈ {20,22,24,26,28}、4 种浮点、VQC vs Yao
#
# 用法：julia --project=... --threads=<N> perf.jl <tag>
#   （tag 用于区分线程配置）
#
# 随机酉门（2 / 3 / 4 比特）作用到大态向量，位置组合覆盖：
#   * low   ：最低 nb 个连续位（批打包最优情形）
#   * mid   ：中间 nb 个连续位
#   * high  ：最高 nb 个连续位
#   * spread：均匀分散在 1..n（gather 跨步最大的最坏情形）
#
# kernel batch size 扫描 8 / 16 / 32 / 64（进程内 `set_kernel_batch!`
# 切换，不同取值各自编译专门版本）。Yao 与 batch size 无关，
# 只在第一个 batch size 下计时。大数组按系统内存自适应跳过 Yao。

include("common.jl")

tag = length(ARGS) >= 1 ? ARGS[1] : string(Threads.nthreads())
dir = ensure_results_dir()
path = joinpath(dir, "perf_t$tag.csv")
NS = (20, 22, 24, 26, 28)
BATCHSIZES = (8, 16, 32, 64)

"门位置组合：连续低 / 中 / 高 + 均匀分散（均已排序去重）。"
function locsets(n::Int, nb::Int)
    low = Tuple(1:nb)
    high = Tuple(n-nb+1:n)
    mid0 = (n - nb) ÷ 2 + 1
    mid = Tuple(mid0:mid0+nb-1)
    spread = [round(Int, 1 + (n - 1) * i / (nb - 1)) for i in 0:nb-1]
    sets = Tuple{String,Tuple{Vararg{Int}}}[("low", low), ("mid", mid), ("high", high)]
    sp = sort(unique(spread))
    sp == collect(low) || push!(sets, ("spread", Tuple(sp)))
    return sets
end

println("="^72)
println("perf: VQC vs Yao（线程 = ", Threads.nthreads(),
        "，batch size = ", BATCHSIZES, "，结果写入 $path）")
println("="^72)

open(path, "w") do io
    println(io, "n,eltype,nb,locs,bs,threads,time_ms,impl")
    for K in BATCHSIZES
        VQC.set_kernel_batch!(K)
        yao_this_k = (K == first(BATCHSIZES))   # Yao 与 batch size 无关
        for n in NS
            for T in TYPES
                bytes = (1 << n) * sizeof(T)
                yao_ok = yao_this_k && 3 * bytes < Sys.total_memory()
                @printf("bs = %2d  n = %2d  %-10s  (%.2f GiB, Yao %s)\n",
                        K, n, T, bytes / 2^30, yao_ok ? "on" : "skipped")
                for nb in (2, 3, 4)
                    D = 1 << nb
                    U = rand_unitary(T, D)
                    U_yao = msb_to_lsb(U, nb)
                    for (loctag, locs) in locsets(n, nb)
                        # VQC（当前 batch size）
                        v = randn(T, 1 << n)
                        sv = StateVector(v, n)
                        t_vqc = bench_min(() -> apply!(sv, U, locs))
                        println(io, "$n,$T,$nb,$loctag,$K,$(Threads.nthreads()),$(t_vqc * 1e3),VQC")
                        @printf("  %d-qubit %-6s  VQC   %10.3f ms\n", nb, loctag, t_vqc * 1e3)
                        # Yao（仅第一个 batch size）
                        if yao_ok
                            reg = ArrayReg(randn(T, 1 << n))
                            t_yao = bench_min(() -> Yao.apply!(reg,
                                put(n, (sort(collect(locs))...,) => matblock(U_yao))))
                            println(io, "$n,$T,$nb,$loctag,$K,$(Threads.nthreads()),$(t_yao * 1e3),Yao")
                            @printf("  %d-qubit %-6s  Yao   %10.3f ms\n", nb, loctag, t_yao * 1e3)
                        end
                    end
                end
            end
            GC.gc()
        end
    end
end
println("完成。")
