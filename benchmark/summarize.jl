# summarize.jl — 汇总 results/ 下的 CSV，打印关键对比表（由 run.jl 调用）
#
# CSV 格式：
#   accuracy: n,eltype,bs,threads,err[,maxnb]
#   perf:     n,eltype,nb,locs,bs,threads,time_ms,impl

using Printf

dir = joinpath(@__DIR__, "results")
isdir(dir) || (println("无 results 目录"); exit(0))
csvs = sort(filter(f -> endswith(f, ".csv"), readdir(dir; join = true)))
isempty(csvs) && (println("无结果文件"); exit(0))

acc_rows = []
perf_rows = []
for f in csvs
    fname = basename(f)
    kind = startswith(fname, "accuracy") ? :acc : :perf
    threads = match(r"_t(\w+)\.csv", fname).captures[1]
    for (i, line) in enumerate(eachline(f))
        i == 1 && continue
        fields = split(line, ",")
        if kind == :acc
            push!(acc_rows, (threads = threads,
                             n = parse(Int, fields[1]), T = fields[2],
                             bs = parse(Int, fields[3]), err = parse(Float64, fields[5])))
        else
            push!(perf_rows, (threads = threads,
                              n = parse(Int, fields[1]), T = fields[2],
                              nb = parse(Int, fields[3]), locs = fields[4],
                              bs = parse(Int, fields[5]), ms = parse(Float64, fields[7]),
                              impl = fields[8]))
        end
    end
end

# ── 精度汇总：每个 (threads, eltype, bs) 的最大误差 ─────────────────────────
println("="^76)
println("accuracy 汇总（VQC vs Yao，最大相对误差）")
println("="^76)
@printf("%-8s %-12s %-6s %-12s %s\n", "threads", "eltype", "bs", "max err", "n range")
accs = Dict{Tuple{String,String,Int},Tuple{Float64,Int,Int}}()
for r in acc_rows
    key = (r.threads, r.T, r.bs)
    cur = get(accs, key, (0.0, r.n, r.n))
    accs[key] = (max(cur[1], r.err), min(cur[2], r.n), max(cur[3], r.n))
end
for key in sort(collect(keys(accs)); by = k -> (parse(Int, k[1]), string(k[2]), k[3]))
    e, nlo, nhi = accs[key]
    @printf("%-8s %-12s %-6d %-12.3e n = %d..%d\n", key[1], key[2], key[3], e, nlo, nhi)
end

# ── 性能汇总 1：batch size 敏感性（ComplexF64，最高线程配置）────────────────
isempty(perf_rows) && (println(); exit(0))
max_thr = maximum(parse(Int, r.threads) for r in perf_rows)
println()
println("="^76)
println("perf：batch size 敏感性（ComplexF64，threads = $(max_thr)，毫秒）")
println("="^76)
@printf("%-4s %-2s %-7s %10s %10s %10s %10s\n", "n", "nb", "locs", "bs=8", "bs=16", "bs=32", "bs=64")
sel = [(r.n, r.nb, r.locs, r.bs) => r.ms for r in perf_rows
       if r.T == "ComplexF64" && parse(Int, r.threads) == max_thr && r.impl == "VQC"]
pdict = Dict(sel)
for n in sort(unique(r.n for r in perf_rows)), nb in (2, 3, 4), locs in ("low", "mid", "high", "spread")
    haskey(pdict, (n, nb, locs, 8)) || continue
    vals = [get(pdict, (n, nb, locs, bs), NaN) for bs in (8, 16, 32, 64)]
    @printf("%-4d %-2d %-7s %10.2f %10.2f %10.2f %10.2f\n", n, nb, locs, vals...)
end

# ── 性能汇总 2：线程 scaling（bs = 16）与 VQC/Yao 对比 ─────────────────────
println()
println("="^76)
println("perf：线程 scaling（bs = 16，ComplexF64，毫秒；括号内为相对 1 线程加速比）")
println("="^76)
thr_list = sort(unique(parse(Int, r.threads) for r in perf_rows))
sdict = Dict{Tuple{Int,Int,String,String,String},Dict{Int,Float64}}()
for r in perf_rows
    (r.bs == 16 && r.T == "ComplexF64") || continue
    key = (r.n, r.nb, r.locs, r.T, r.impl)
    d = get!(sdict, key, Dict{Int,Float64}())
    d[parse(Int, r.threads)] = r.ms
end
skeys = sort(collect(keys(sdict)); by = k -> (k[1], k[2], k[3], k[5]))
last_n = -1
for key in skeys
    n, nb, locs, T, impl = key
    n != last_n && (println(); @printf("n = %d\n", n); global last_n = n)
    d = sdict[key]
    base = get(d, 1, NaN)
    parts = map(thr_list) do thr
        ms = get(d, thr, NaN)
        speed = base > 0 ? base / ms : NaN
        @sprintf("t%d:%9.2f (x%.1f)", thr, ms, speed)
    end
    @printf("  %d-qubit %-6s [%s]  %s\n", nb, locs, impl, join(parts, "  "))
end
println()
println("完整数据见 results/*.csv。")
