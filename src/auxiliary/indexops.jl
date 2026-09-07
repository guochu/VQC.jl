# indexops.jl — 比特索引位操作（小端序，0-based）

"""
    _bit_insert_zeros(r, key) -> Int

把缩减索引 `r`（已在 `key` 对应比特位置上清零的索引）的比特重新插入到
`key`（LSB-first 的 0-based 比特位置元组）指定的位置上，其余位置为 0。
"""
function _bit_insert_zeros(r::Int, key::Tuple{Vararg{Int}})
    x = r
    @inbounds for q in key
        low = (1 << q) - 1
        x = (x & low) | ((x & ~low) << 1)
    end
    return x
end

"""
    _offsets(key) -> NTuple{2^N, Int}

局域基矢索引 `j`（LSB-first，`j = Σ b_i 2^(i-1)`，最低位对应 `key[1]`）
对应的全局振幅偏移（`offsets[j+1] = Σ b_i 2^(key[i])`）。
"""
_offsets(key::Tuple{}) = (0,)
function _offsets(key::Tuple)
    rest = _offsets(Base.tail(key))
    q = 1 << first(key)
    return _interleave(rest, rest .+ q)
end

_interleave(a::Tuple{}, b::Tuple{}) = ()
function _interleave(a::Tuple, b::Tuple)
    return (first(a), first(b), _interleave(Base.tail(a), Base.tail(b))...)
end

"""
    _scatter(l, key) -> Int

把局域索引 `l` 的比特散布到 `key` 指定的位置（等价于 `_offsets(key)[l+1]`，
供运行期动态路径使用）。
"""
function _scatter(l::Int, key::Tuple{Vararg{Int}})
    x = 0
    @inbounds for (i, q) in enumerate(key)
        x |= ((l >> (i - 1)) & 1) << q
    end
    return x
end
