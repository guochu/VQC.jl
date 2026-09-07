# parallel.jl — 轻量并行工具

"并行阈值：缩减迭代数（reduced iterations）低于此值时串行执行。"
const THREAD_MIN = 1 << 12

"""
    _run_range(f, n) -> nothing

把 `[0, n-1]` 的迭代区间（0-based 闭区间）按线程分块后调用
`f(istart, iend)`（两者均为 0-based 闭区间端点）；区间过小或单线程时
直接 `f(0, n-1)`。
"""
function _run_range(f::F, n::Int) where {F}
    nt = Threads.nthreads()
    if n >= THREAD_MIN && nt > 1
        Threads.@threads for t in 1:nt
            istart = div(n * (t - 1), nt)
            iend = div(n * t, nt) - 1
            f(istart, iend)
        end
    else
        f(0, n - 1)
    end
    return nothing
end

"""
    _run_range_sum(f, T, n) -> T

同 `_run_range`，但 `f(istart, iend)` 返回部分和，逐线程求和归并。
"""
function _run_range_sum(f::F, ::Type{T}, n::Int) where {T,F}
    nt = Threads.nthreads()
    if n >= THREAD_MIN && nt > 1
        parts = Vector{T}(undef, nt)
        Threads.@threads for t in 1:nt
            istart = div(n * (t - 1), nt)
            iend = div(n * t, nt) - 1
            parts[t] = f(istart, iend)
        end
        return sum(parts)
    else
        return f(0, n - 1)
    end
end
