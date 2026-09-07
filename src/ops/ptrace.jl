# ptrace.jl — 偏迹（约化密度矩阵，核心层）

"""
    partial_tr(s::StateVector, sites::AbstractVector{Int}) -> DensityMatrix

对 `sites`（1-based）求偏迹；保留比特按升序重新编号。
空 `sites` 返回完整密度矩阵；全比特求迹返回标量 `⟨ψ|ψ⟩`。
"""
function partial_tr(s::StateVector, sites::AbstractVector{Int})
    isempty(sites) && return DensityMatrix(s)
    n = _nqubits(s)
    _check_ptrace_sites(sites, n)
    (length(sites) == n) && return dot(s, s)
    keep = setdiff(1:n, collect(sites))
    nk = length(keep)
    dk, ds = 1 << nk, 1 << length(sites)
    vt = permutedims(reshape(storage(s), ntuple(_ -> 2, n)),
                     vcat(keep, sites))  # permutedims 用 1-based 轴号
    a = reshape(vt, dk, ds)   # 行 = 保留比特（小端序），列 = 求迹比特
    return DensityMatrix(a * a', nk)
end

"""
    partial_tr(s::DensityMatrix, sites::AbstractVector{Int}) -> DensityMatrix

对 `sites`（1-based）求偏迹；保留比特按升序重新编号。
空 `sites` 返回自身副本；全比特求迹返回 `tr(s)`。
"""
function partial_tr(s::DensityMatrix, sites::AbstractVector{Int})
    isempty(sites) && return copy(s)
    n = _nqubits(s)
    _check_ptrace_sites(sites, n)
    (length(sites) == n) && return tr(s)
    keep = setdiff(1:n, collect(sites))
    nk = length(keep)
    nsites = length(sites)
    dk, ds = 1 << nk, 1 << nsites
    vt = permutedims(reshape(s.data, ntuple(_ -> 2, 2n)),
                     vcat(keep, sites, keep .+ n, sites .+ n))
    vt = reshape(vt, dk, ds, dk, ds)
    out = zeros(eltype(vt), dk, dk)
    for r in 1:ds
        out .+= @view vt[:, r, :, r]   # 对 (r_sites, c_sites) 求对角和
    end
    return DensityMatrix(out, nk)
end

partial_tr(s::Union{StateVector,DensityMatrix}, sites::Int...) = partial_tr(s, collect(sites))

function _check_ptrace_sites(sites, n::Int)
    length(unique(sites)) == length(sites) || throw(ArgumentError("duplicate sites not allowed"))
    all(q -> 1 <= q <= n, sites) || throw(ArgumentError("site out of range [1, $n]"))
    return nothing
end
