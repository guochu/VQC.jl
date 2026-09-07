# tensorops.jl — 张量辅助（比特交换 / 熵）

"""
    swap!(v::AbstractVector, i::Int, j::Int)

就地交换 1-based 比特 `i` 与 `j`（把 `v` 视为 2^n 振幅向量）。
"""
function swap!(v::AbstractVector, i::Int, j::Int)
    (i == j) && return v
    i > j && return swap!(v, j, i)
    i -= 1
    j -= 1
    fsize = 1 << i
    msize = 1 << (j - i - 1)
    bsize = div(length(v), fsize * 4 * msize)
    s = reshape(v, (fsize, 2, msize, 2, bsize))
    tmp = s[:, 1, :, 2, :]
    s[:, 1, :, 2, :] = s[:, 2, :, 1, :]
    s[:, 2, :, 1, :] = tmp
    return v
end

"""
    entropy(p::AbstractVector{<:Real}; tol=1e-12)

概率分布的香农熵（以 2 为底）。
"""
function entropy(v::AbstractVector{<:Real}; tol::Real=1e-12)
    s = sum(v)
    isapprox(s, 1.0; atol=1e-8) || throw(ArgumentError("input is not a probability distribution (sum = $s)"))
    acc = 0.0
    for item in v
        item <= tol && continue
        acc -= item * log2(item)
    end
    return acc
end

"""
    renyi_entropy(p; α=2)

概率分布的 Rényi 熵（以 2 为底）；`α = 1` 退化为香农熵。
"""
function renyi_entropy(v::AbstractVector{<:Real}; α::Real=2)
    α == 1 && return entropy(v)
    α > 0 || throw(ArgumentError("α must be positive"))
    acc = zero(float(eltype(v)))
    for item in v
        item > 0 || continue
        acc += item^α
    end
    return 1 / (1 - α) * log2(acc)
end
