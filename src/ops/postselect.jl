# postselect.jl — 后选择（核心层）

"""
    post_select!(s::Union{StateVector,DensityMatrix}, q::Integer, val::Integer=0) -> Float64

把 qubit `q` 后选择到 `val`（就地投影并归一化），返回发生概率。
"""
function post_select!(s::Union{StateVector,DensityMatrix}, q::Integer, val::Integer=0)
    q = Int(q)
    val = Int(val)
    n = _nqubits(s)
    1 <= q <= n || throw(ArgumentError("qubit index $q out of range [1, $n]"))
    val in (0, 1) || throw(ArgumentError("post-selection value must be 0 or 1"))
    p0, p1 = marginal_probabilities(s, q)
    p = val == 0 ? p0 : p1
    p > 0 || throw(ArgumentError("post-selection probability is zero"))
    _project!(s, q - 1, val, p)
    return p
end

"""
    post_select(s::Union{StateVector,DensityMatrix}, q::Integer, val::Integer=0) -> (s', p)

非就地后选择：返回坍缩后的新态与发生概率。
"""
function post_select(s::Union{StateVector,DensityMatrix}, q::Integer, val::Integer=0)
    out = copy(s)
    p = post_select!(out, q, val)
    return out, p
end
