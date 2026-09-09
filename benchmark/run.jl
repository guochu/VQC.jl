# run.jl — benchmark 总调度
#
# 用法（在本目录下）：
#   julia --project=... run.jl            # 精度 + 性能全跑
#   julia --project=... run.jl accuracy   # 只跑精度
#   julia --project=... run.jl perf       # 只跑性能
#
# 精度测试（accuracy.jl）：n = 1..20、4 种 BlasFloat、batch size
# 8/16/32/64，在 1/2/3/4 线程下各跑一遍（最多不超过 CPU 核数），
# VQC 与 Yao 对拍；线路含 1..5 比特随机酉门、多位置（见 accuracy.jl）。
#
# 性能测试（perf.jl）：n ∈ {20,22,24,26,28}、4 种 BlasFloat、
# 2/3/4-qubit 随机酉门 × 4 组位置（low/mid/high/spread）、
# batch size 8/16/32/64（进程内切换）；线程数取 1 与 8（不超过
# CPU 核数）两个代表端点（Yao 与 batch size 无关，每组线程只计一次）。
#
# 各配置以子进程方式运行（线程数是进程级参数），结果 CSV 写入
# results/，最后用 summarize.jl 汇总打印。

using Printf

dir = @__DIR__
mode = get(ARGS, 1, "all")
ncores = Sys.CPU_THREADS

acc_threads = unique(filter(t -> t <= ncores, 1:4))   # 精度：1..4 线程
perf_threads = unique(filter(t -> t <= ncores, (1, 8)))

function spawn(script::String, threads::Int)
    println("#"^72)
    @printf("运行 %s（--threads=%d）\n", script, threads)
    println("#"^72)
    run(`$(Base.julia_cmd()) --project=$(Base.active_project()) --threads=$threads $(joinpath(dir, script)) $threads`)
end

if mode in ("accuracy", "all")
    for t in acc_threads
        spawn("accuracy.jl", t)
    end
end
if mode in ("perf", "all")
    for t in perf_threads
        spawn("perf.jl", t)
    end
end

println()
include("summarize.jl")
