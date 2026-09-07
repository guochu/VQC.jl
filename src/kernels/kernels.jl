# kernels.jl — 底层计算核：局域酉算子作用与期望值

# 约定：
#   * `key::NTuple{N,Int}` 为 **LSB-first、0-based** 的比特位置元组；
#   * 局域矩阵 `U` 的行列索引按 `key` 顺序小端展开
#     （即 `key[1]` 对应矩阵索引的最低位）；
#   * 由 `GateOp.qubits`（MSB-first）转换：`key = reverse(qs)`；
#   * `_bit_insert_zeros` 的逐步插入算法要求升序处理，故 `base` 计算使用
#     排序后的 `skey`（集合相同，顺序无关）；`offs` 仍用原 `key`（决定矩阵位序）。

_sorted_key(key::Tuple{Vararg{Int}}) = Tuple(sort!(collect(Int, key)))

"""
    apply_kernel!(v::AbstractVector, key::NTuple{N,Int}, U::AbstractMatrix) -> v

把 `N` 比特局域矩阵 `U` 就地作用到振幅向量 `v` 上（`key` LSB-first、0-based）。
"""
function apply_kernel!(v::AbstractVector, key::NTuple{N,Int}, U::AbstractMatrix) where {N}
    D = 1 << N
    (size(U, 1) == size(U, 2) == D) ||
        throw(ArgumentError("matrix size $(size(U)) does not match $N qubit(s)"))
    nred = length(v) >> N
    skey = _sorted_key(key)
    if N <= 4
        Us = SMatrix{D,D,eltype(v)}(U)
        offs = _offsets(key)
        _run_range(nred) do istart, iend
            _apply_static_range!(v, Us, skey, offs, istart, iend)
        end
    else
        offs = collect(Int, _offsets(key))
        _run_range(nred) do istart, iend
            _apply_dynamic_range!(v, U, skey, offs, istart, iend)
        end
    end
    return v
end

function _apply_static_range!(v::AbstractVector{T}, U::SMatrix{D,D},
                              skey::NTuple{N,Int}, offs::NTuple{D,Int},
                              istart::Int, iend::Int) where {T,N,D}
    @inbounds for r in istart:iend
        base = _bit_insert_zeros(r, skey)
        loc = SVector{D,T}(ntuple(i -> v[base + offs[i] + 1], Val(D)))
        out = U * loc
        for i in 1:D
            v[base + offs[i] + 1] = out[i]
        end
    end
    return nothing
end

function _apply_dynamic_range!(v::AbstractVector, U::AbstractMatrix,
                               skey::Tuple{Vararg{Int}}, offs::Vector{Int},
                               istart::Int, iend::Int)
    D = length(offs)
    T = eltype(v)
    loc = Vector{T}(undef, D)
    out = Vector{T}(undef, D)
    @inbounds for r in istart:iend
        base = _bit_insert_zeros(r, skey)
        for j in 1:D
            loc[j] = v[base + offs[j] + 1]
        end
        for i in 1:D
            acc = zero(T)
            @simd for j in 1:D
                acc += U[i, j] * loc[j]
            end
            out[i] = acc
        end
        for i in 1:D
            v[base + offs[i] + 1] = out[i]
        end
    end
    return nothing
end

"""
    expect_kernel(v::AbstractVector, key::NTuple{N,Int}, U::AbstractMatrix) -> scalar

计算局域矩阵元 `⟨v|U|v⟩`。
"""
function expect_kernel(v::AbstractVector, key::NTuple{N,Int}, U::AbstractMatrix) where {N}
    D = 1 << N
    (size(U, 1) == size(U, 2) == D) ||
        throw(ArgumentError("matrix size $(size(U)) does not match $N qubit(s)"))
    OT = promote_type(eltype(v), eltype(U))
    nred = length(v) >> N
    skey = _sorted_key(key)
    if N <= 4
        Us = SMatrix{D,D,eltype(v)}(U)
        offs = _offsets(key)
        return _run_range_sum(OT, nred) do istart, iend
            _expect_static_range(v, Us, skey, offs, istart, iend)
        end
    else
        offs = collect(Int, _offsets(key))
        return _run_range_sum(OT, nred) do istart, iend
            _expect_dynamic_range(v, U, skey, offs, istart, iend)
        end
    end
end

"""
    expect_kernel(vc::AbstractVector, v::AbstractVector, key, U) -> scalar

计算局域矩阵元 `⟨vc|U|v⟩`。
"""
function expect_kernel(vc::AbstractVector, v::AbstractVector, key::NTuple{N,Int}, U::AbstractMatrix) where {N}
    length(vc) == length(v) || throw(DimensionMismatch("state length mismatch"))
    D = 1 << N
    (size(U, 1) == size(U, 2) == D) ||
        throw(ArgumentError("matrix size $(size(U)) does not match $N qubit(s)"))
    OT = promote_type(eltype(vc), eltype(v), eltype(U))
    nred = length(v) >> N
    skey = _sorted_key(key)
    if N <= 4
        Us = SMatrix{D,D,eltype(v)}(U)
        offs = _offsets(key)
        return _run_range_sum(OT, nred) do istart, iend
            _expect2_static_range(vc, v, Us, skey, offs, istart, iend)
        end
    else
        offs = collect(Int, _offsets(key))
        return _run_range_sum(OT, nred) do istart, iend
            _expect2_dynamic_range(vc, v, U, skey, offs, istart, iend)
        end
    end
end

function _expect_static_range(v::AbstractVector{T}, U::SMatrix{D,D},
                              skey::NTuple{N,Int}, offs::NTuple{D,Int},
                              istart::Int, iend::Int) where {T,N,D}
    acc = zero(promote_type(T, eltype(U)))
    @inbounds for r in istart:iend
        base = _bit_insert_zeros(r, skey)
        loc = SVector{D,T}(ntuple(i -> v[base + offs[i] + 1], Val(D)))
        acc += dot(loc, U, loc)
    end
    return acc
end

function _expect2_static_range(vc::AbstractVector, v::AbstractVector{T}, U::SMatrix{D,D},
                               skey::NTuple{N,Int}, offs::NTuple{D,Int},
                               istart::Int, iend::Int) where {T,N,D}
    acc = zero(promote_type(eltype(vc), T, eltype(U)))
    @inbounds for r in istart:iend
        base = _bit_insert_zeros(r, skey)
        loc = SVector{D,T}(ntuple(i -> v[base + offs[i] + 1], Val(D)))
        loc_c = SVector{D,eltype(vc)}(ntuple(i -> vc[base + offs[i] + 1], Val(D)))
        acc += dot(loc_c, U, loc)
    end
    return acc
end

function _expect_dynamic_range(v::AbstractVector, U::AbstractMatrix,
                               skey::Tuple{Vararg{Int}}, offs::Vector{Int},
                               istart::Int, iend::Int)
    D = length(offs)
    T = eltype(v)
    OT = promote_type(T, eltype(U))
    loc = Vector{T}(undef, D)
    acc = zero(OT)
    @inbounds for r in istart:iend
        base = _bit_insert_zeros(r, skey)
        for j in 1:D
            loc[j] = v[base + offs[j] + 1]
        end
        for i in 1:D
            acc_i = zero(OT)
            @simd for j in 1:D
                acc_i += conj(loc[i]) * U[i, j] * loc[j]
            end
            acc += acc_i
        end
    end
    return acc
end

function _expect2_dynamic_range(vc::AbstractVector, v::AbstractVector, U::AbstractMatrix,
                                skey::Tuple{Vararg{Int}}, offs::Vector{Int},
                                istart::Int, iend::Int)
    D = length(offs)
    OT = promote_type(eltype(vc), eltype(v), eltype(U))
    loc = Vector{eltype(v)}(undef, D)
    loc_c = Vector{eltype(vc)}(undef, D)
    acc = zero(OT)
    @inbounds for r in istart:iend
        base = _bit_insert_zeros(r, skey)
        for j in 1:D
            loc[j] = v[base + offs[j] + 1]
            loc_c[j] = vc[base + offs[j] + 1]
        end
        for i in 1:D
            acc_i = zero(OT)
            @simd for j in 1:D
                acc_i += conj(loc_c[i]) * U[i, j] * loc[j]
            end
            acc += acc_i
        end
    end
    return acc
end

"""
    dm_expect_kernel(rho::AbstractVector, d::Int, key::NTuple{N,Int}, m::AbstractMatrix) -> scalar

计算 `tr(ρ m)`，其中 `rho` 是长度 `d²`（`d = 2^n`）的列 major 平坦存储，
`key`/`m` 为 `m` 的局域比特与局域矩阵。
"""
function dm_expect_kernel(rho::AbstractVector, d::Int, key::NTuple{N,Int}, m::AbstractMatrix) where {N}
    D = 1 << N
    (size(m, 1) == size(m, 2) == D) ||
        throw(ArgumentError("matrix size $(size(m)) does not match $N qubit(s)"))
    (d >> N) >= 0 || throw(ArgumentError("state too small"))
    OT = promote_type(eltype(rho), eltype(m))
    mv = vec(transpose(m))   # 与 ρ 局域块 (row + D*col) 的收集布局对齐
    nred = d >> N
    skey = _sorted_key(key)
    if N <= 4
        mv_s = SVector{D * D,eltype(m)}(mv)
        offs = _offsets(key)
        return _run_range_sum(OT, nred) do istart, iend
            _dm_expect_static_range(rho, d, mv_s, skey, offs, istart, iend)
        end
    else
        offs = collect(Int, _offsets(key))
        mvv = Vector{eltype(m)}(mv)
        return _run_range_sum(OT, nred) do istart, iend
            _dm_expect_dynamic_range(rho, d, mvv, skey, offs, istart, iend)
        end
    end
end

function _dm_expect_static_range(rho::AbstractVector{T}, d::Int, mv::SVector{D2},
                                 skey::NTuple{N,Int}, offs::NTuple{D,Int},
                                 istart::Int, iend::Int) where {T,N,D,D2}
    acc = zero(promote_type(T, eltype(mv)))
    @inbounds for rest in istart:iend
        base = _bit_insert_zeros(rest, skey)
        loc = SVector{D2,T}(ntuple(i -> begin
            k = i - 1
            lj, li = divrem(k, D)      # k = li + D*lj（列 major：li 为行局域索引）
            ri = base + offs[li+1]
            ci = base + offs[lj+1]
            rho[ri + d * ci + 1]
        end, Val(D2)))
        acc += sum(loc .* mv)
    end
    return acc
end

function _dm_expect_dynamic_range(rho::AbstractVector, d::Int, mv::Vector,
                                  skey::Tuple{Vararg{Int}}, offs::Vector{Int},
                                  istart::Int, iend::Int)
    D = length(offs)
    OT = promote_type(eltype(rho), eltype(mv))
    acc = zero(OT)
    @inbounds for rest in istart:iend
        base = _bit_insert_zeros(rest, skey)
        acc_k = zero(OT)
        for i in 1:D
            k = i - 1
            lj, li = divrem(k, D)
            ri = base + offs[li+1]
            ci = base + offs[lj+1]
            acc_k += rho[ri + d * ci + 1] * mv[i]
        end
        acc += acc_k
    end
    return acc
end
