# legacy_kernels.jl — 原始 VQC（save/VQC）kernel 的原样移植，供 benchmark 对比。
#
# 来源：save/VQC/src/applygates/generic/{serial_short_range,
# threaded_short_range, threaded_long_range}.jl 与
# save/VQC/src/auxiliary/parallel_for.jl（仅去掉对包类型的依赖）。
# 比特位置约定为 **1-based**。

module Legacy

using Base.Threads
using StaticArrays

const MIN_SIZE = 1024

get_size_1(total_itr::Int, n_threads::Int, thread_id::Int) = div(total_itr * thread_id, n_threads)
get_size_2(total_itr::Int, n_threads::Int, thread_id::Int) = div(total_itr * (thread_id + 1), n_threads)

function parallel_run(total_itr::Int, n_threads::Int, f::Function, args...)
    if (total_itr >= MIN_SIZE) && (n_threads > 1)
        Threads.@threads for thread_id in 0:(n_threads-1)
            ist = get_size_1(total_itr, n_threads, thread_id)
            ifn = get_size_2(total_itr, n_threads, thread_id) - 1
            f(ist, ifn, args...)
        end
    else
        f(0, total_itr-1, args...)
    end
end

include("legacy/serial_short_range.jl")
include("legacy/threaded_short_range.jl")
include("legacy/threaded_long_range.jl")

"原始 VQC 的线程化入口：`key` 为 1-based 位置元组（升序）。"
apply_gate_threaded!(key::Tuple, U::AbstractMatrix, v::AbstractVector) =
    _apply_gate_threaded2!(key, U, v)

end # module Legacy
