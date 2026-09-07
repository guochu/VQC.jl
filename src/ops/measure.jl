# measure.jl — 计算基测量 / 采样（核心层）

"""
    probabilities(s::Union{StateVector,DensityMatrix}) -> Vector{Float64}

全部计算基概率（长度 `2^n`，索引 = 基矢整数值，小端序）。
"""
probabilities(s::StateVector) = abs2.(storage(s))

function probabilities(s::DensityMatrix)
    d = 1 << _nqubits(s)
    return [real(s.data[i * d + i + 1]) for i in 0:d-1]
end

"""
    probabilities(s, qubits::AbstractVector{Int}) -> Vector{Float64}

指定比特子集的边缘概率分布：长度 `2^k`（`k = length(qubits)`），
输出的第 `j` 位（LSB-first）对应 `qubits[j]` 的取值。
"""
function probabilities(s::Union{StateVector,DensityMatrix}, qs::AbstractVector{Int})
    n = _nqubits(s)
    k = length(qs)
    all(q -> 1 <= q <= n, qs) || throw(ArgumentError("qubit index out of range [1, $n]"))
    length(unique(qs)) == k || throw(ArgumentError("duplicate qubits"))
    out = zeros(Float64, 1 << k)
    ps = probabilities(s)
    d = length(ps)
    for idx in 0:d-1
        j = 0
        for t in 1:k
            j |= ((idx >> (qs[t] - 1)) & 1) << (t - 1)
        end
        out[j+1] += ps[idx+1]
    end
    return out
end

"单比特边缘概率 `[P(0), P(1)]`。"
probabilities(s::Union{StateVector,DensityMatrix}, q::Int) = probabilities(s, [q])

"""
    measure!(s::Union{StateVector,DensityMatrix}, q::Integer) -> (outcome, p)

测量 qubit `q`（1-based）并就地坍缩，返回 `(outcome, p)`：测量结果
（0/1）及其概率。
"""
function measure!(s::Union{StateVector,DensityMatrix}, q::Integer)
    q = Int(q)
    n = _nqubits(s)
    1 <= q <= n || throw(ArgumentError("qubit index $q out of range [1, $n]"))
    p0, p1 = probabilities(s, q)
    outcome = rand() * (p0 + p1) < p0 ? 0 : 1
    p = outcome == 0 ? p0 : p1
    p > 0 || throw(ArgumentError("cannot collapse onto zero-probability outcome"))
    _project!(s, q - 1, outcome, p)
    return outcome, p
end

"""
    measure!(s, qs::AbstractVector{Int}) -> Vector{Tuple{Int,Float64}}

依次测量一组比特（就地坍缩），返回 `(outcome, p)` 元组的向量。
"""
function measure!(s::Union{StateVector,DensityMatrix}, qs::AbstractVector{Int})
    return Tuple{Int,Float64}[measure!(s, q) for q in qs]
end

"""
    sample(s::Union{StateVector,DensityMatrix}, shots::Int) -> Dict{Int,Int}
    sample(s, qubits::AbstractVector{Int}, shots::Int) -> Dict{Int,Int}

按计算基采样 `shots` 次，返回 `Dict{Int,Int}`：键为（子集）比特串的
整数值（`qubits[1]` = 最低位；全集采样时 = 基矢索引），值为出现次数。
"""
function sample(s::Union{StateVector,DensityMatrix}, qs::AbstractVector{Int}, shots::Int)
    shots >= 0 || throw(ArgumentError("shots must be non-negative"))
    ps = probabilities(s, qs)
    return _sample_counts(ps, shots)
end

function sample(s::Union{StateVector,DensityMatrix}, shots::Int)
    return sample(s, collect(1:_nqubits(s)), shots)
end

function _sample_counts(ps::AbstractVector{Float64}, shots::Int)
    cum = cumsum(ps)
    counts = Dict{Int,Int}()
    for _ in 1:shots
        i = searchsortedfirst(cum, rand())
        i = min(i, length(cum))
        counts[i-1] = get(counts, i-1, 0) + 1
    end
    return counts
end

# ── 投影（测量坍缩与后选择共用） ─────────────────────────────────────────────

"把 `q` 投影到 `outcome` 并除以 `sqrt(p)` 归一化（就地）。"
function _project!(s::StateVector, q::Int, outcome::Int, p::Real)
    P = outcome == 0 ? SMatrix{2,2}(1.0, 0.0, 0.0, 0.0) : SMatrix{2,2}(0.0, 0.0, 0.0, 1.0)
    apply_kernel!(storage(s), (q,), P)
    rmul!(storage(s), 1 / sqrt(p))
    return s
end

function _project!(s::DensityMatrix, q::Int, outcome::Int, p::Real)
    P = outcome == 0 ? SMatrix{2,2}(1.0, 0.0, 0.0, 0.0) : SMatrix{2,2}(0.0, 0.0, 0.0, 1.0)
    n = _nqubits(s)
    apply_kernel!(s.data, (q,), P)
    apply_kernel!(s.data, (q + n,), P)  # conj(P) = P
    rmul!(s.data, 1 / p)
    return s
end
