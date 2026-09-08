# kernels.jl — 底层计算核：局域酉算子作用与期望值
#
# 性能设计（对齐原始 VQC 的思路，用批打包通用化）：
#   * 门的比特数 `N` 由 `key::NTuple{N,Int}` 在**编译期**给出——
#     不存在运行期动态维度，所有循环界均为静态；
#   * 门恰占据**最低 N 个连续位**时（`skey == (0,…,N-1)`，最常见情形），
#     每个局域块是连续的 `2^N` 个振幅——一次 gathers `K` 个连续块打包为
#     `SMatrix{D,K}`，用一次 `U * V` 批量完成（内存连续、SIMD 友好）；
#   * 其它位置布局走逐块静态展开（正确性优先，gather 跨步固定）；
#   * 期望值核同理：`Σₖ ⟨vₖ|U|vₖ⟩ = sum(V .* (U * V))`；
#   * 多线程由 `_run_range` / `_run_range_sum` 按批分块。

# 约定：
#   * `key::NTuple{N,Int}` 为 **LSB-first、0-based** 的比特位置元组；
#   * 局域矩阵 `U` 的行列索引按 `key` 顺序小端展开
#     （即 `key[1]` 对应矩阵索引的最低位）；
#   * 由 `GateOp.qubits`（MSB-first）转换：`key = _lsb_key(qs)`；
#   * `_bit_insert_zeros` 的逐步插入算法要求升序处理，故 `base` 计算使用
#     排序后的 `skey`（集合相同，顺序无关）；`offs` 仍用原 `key`（决定矩阵位序）。

_sorted_key(key::NTuple{N,Int}) where {N} = Tuple(sort!(collect(Int, key)))

# 批处理块数（全局可调）：门在最低连续位时，一次 gathers `K` 个连续块
# 打包为 `SMatrix{D,K}` 批量乘。现代机器缓存较大，可按硬件调大
# （16 / 32 / 64）；不同取值会各自编译专门的 kernel 版本（`Val{K}` 常量化）。
const _KERNEL_BATCH = Ref(16)

"""
    kernel_batch() -> Int

查询 kernel 批处理的块数（每批 gather 的局域块个数）。
"""
kernel_batch() = _KERNEL_BATCH[]

"""
    set_kernel_batch!(n::Integer) -> Int

设置 kernel 批处理的块数（默认 16，须为 2 的幂）。缓存较大时可调大
（如 32 / 64）；取值越大单批的寄存器 / 编译产物越大，过大反而可能因
寄存器溢出变慢，建议按硬件实测选择。下一次 kernel 调用即生效。
"""
function set_kernel_batch!(n::Integer)
    n >= 1 || throw(ArgumentError("batch size must be positive"))
    ispow2(n) || throw(ArgumentError("batch size must be a power of 2, got $n"))
    _KERNEL_BATCH[] = Int(n)
    return Int(n)
end

# 基础位操作工具（`_bit_insert_zeros` / `_offsets` / `_interleave` / `_scatter`）
# 定义见 `auxiliary/indexops.jl`。

# ── apply：局域酉算子作用 ────────────────────────────────────────────────────

"""
    apply_kernel!(v::AbstractVector, key::NTuple{N,Int}, U::AbstractMatrix) -> v

把 `N` 比特局域矩阵 `U` 就地作用到振幅向量 `v` 上（`key` LSB-first、0-based）。
`U` 被转换到 `eltype(v)` 后作用；门位于最低连续位时走连续批打包路径。
"""
function apply_kernel!(v::AbstractVector, key::NTuple{N,Int}, U::AbstractMatrix) where {N}
    D = 1 << N
    (size(U, 1) == size(U, 2) == D) ||
        throw(ArgumentError("matrix size $(size(U)) does not match $N qubit(s)"))
    nred = length(v) >> N
    skey = _sorted_key(key)
    offs = _offsets(key)
    Uc = SMatrix{D,D,eltype(v)}(U)
    if offs == ntuple(j -> j - 1, Val(D))
        # 门恰为最低 N 个连续位：块连续，按批打包
        K = min(_KERNEL_BATCH[], D)
        nchunk, ntail = divrem(nred, K)
        nchunk > 0 && _run_range(nchunk) do a, b
            _apply_batched_range!(v, Uc, a, b, Val(K), Val(N))
        end
        ntail > 0 && _apply_single_range!(v, Uc, skey, offs,
                                          nchunk * K, nred - 1)
    else
        _run_range(nred) do a, b
            _apply_single_range!(v, Uc, skey, offs, a, b)
        end
    end
    return v
end

"`K` 个连续块（每块 `D = 2^N` 个连续振幅）打包为 `SMatrix{D,K}` 一次乘完。"
function _apply_batched_range!(v::AbstractVector{T}, U::SMatrix{D,D,T},
                               istart::Int, iend::Int, ::Val{K}, ::Val{N}) where {T,D,K,N}
    len = D * K
    @inbounds for t in istart:iend
        base0 = len * t
        V = SMatrix{D,K,T}(ntuple(j -> v[base0 + j], Val(len)))
        O = U * V
        for j in 1:len
            v[base0 + j] = O[j]
        end
    end
    return nothing
end

"逐块静态展开（任意位置布局；批处理尾部与高位门共用）。"
function _apply_single_range!(v::AbstractVector{T}, U::SMatrix{D,D,T},
                              skey::NTuple{N,Int}, offs::NTuple{D,Int},
                              istart::Int, iend::Int) where {T,D,N}
    @inbounds for rest in istart:iend
        base = _bit_insert_zeros(rest, skey)
        loc = SVector{D,T}(ntuple(i -> v[base + offs[i] + 1], Val(D)))
        out = U * loc
        for i in 1:D
            v[base + offs[i] + 1] = out[i]
        end
    end
    return nothing
end

# ── apply_kernel_dm：密度矩阵平坦存储上的局域算子作用 ────────────────────────

"""
    apply_kernel_dm!(rho::AbstractVector, d::Int, key::NTuple{N,Int}, m::AbstractMatrix) -> rho

把 `N` 比特局域**超算子** `m`（`D²×D²`，`D = 2^N`）就地作用到密度矩阵的
平坦存储 `rho`（长度 `d²`，列 major）上：每个局域块 `ρ₍ᵢⱼ₎`（`D²` 个元素，
`ρ₍ᵢⱼ₎ = ρ[b+ᵢ, b+ⱼ]`，`b` 为块基址）被 `m · vec(ρ₍ᵢⱼ₎)` 替换。供 Kraus
等局域量子信道使用（`m = Σₖ kron(conj(kₖ), kₖ)`）。

`m` 的行 / 列索引按 `key` 小端展开（与局域密度块收集布局一致）。
"""
function apply_kernel_dm!(rho::AbstractVector, d::Int, key::NTuple{N,Int}, m::AbstractMatrix) where {N}
    D2 = 1 << (2 * N)
    (size(m, 1) == size(m, 2) == D2) ||
        throw(ArgumentError("matrix size $(size(m)) does not match $N qubit(s)"))
    nred = d >> N                 # 行（或列）非 key 位组合数
    skey = _sorted_key(key)
    offs = _offsets(key)
    Mc = SMatrix{D2,D2,eltype(rho)}(m)
    D = 1 << N
    nblk = nred * nred            # (行基址, 列基址) 独立遍历
    _run_range(nblk) do a, b
        _dm_apply_single_range!(rho, d, Mc, skey, offs, a, b, Val(D), Val(N), nred)
    end
    return rho
end

function _dm_apply_single_range!(rho::AbstractVector{T}, d::Int, m::SMatrix{D2,D2,T},
                                 skey::NTuple{N,Int}, offs::NTuple{D,Int},
                                 istart::Int, iend::Int, ::Val{D}, ::Val{N},
                                 nred::Int) where {T,D,N,D2}
    @inbounds for rest in istart:iend
        br, bc = divrem(rest, nred)   # 行基址块 / 列基址块（相互独立）
        row = _bit_insert_zeros(br, skey)
        col = _bit_insert_zeros(bc, skey)
        loc = SVector{D2,T}(ntuple(i -> begin
            lj, li = divrem(i - 1, D)    # i - 1 = li + D*lj（列 major：li 为行局域索引）
            rho[row + offs[li+1] + d * (col + offs[lj+1]) + 1]
        end, Val(D2)))
        out = m * loc
        for i in 1:D2
            lj, li = divrem(i - 1, D)
            rho[row + offs[li+1] + d * (col + offs[lj+1]) + 1] = out[i]
        end
    end
    return nothing
end

# ── out-of-place：局域算子的 mul! / axpy! 核（y = U·x / y += α·U·x） ──────────
#
# 供 SpinOpTerm / SpinOpSum 的 LinearAlgebra.mul! / axpy! 重载使用：
# 乘积项逐因子链式作用时无需分配中间振幅数组（见 hamiltonian.jl）。

"""
    lmul_kernel!(y::AbstractVector, x::AbstractVector, key, U) -> y

`y = U·x`（`N` 比特局域块逐块 gather，`key` LSB-first、0-based）。
`y` 与 `x` 不得共享存储。
"""
function lmul_kernel!(y::AbstractVector, x::AbstractVector, key::NTuple{N,Int}, U::AbstractMatrix) where {N}
    D = 1 << N
    (size(U, 1) == size(U, 2) == D) ||
        throw(ArgumentError("matrix size $(size(U)) does not match $N qubit(s)"))
    length(y) == length(x) || throw(DimensionMismatch("state length mismatch"))
    OT = promote_type(eltype(y), eltype(x))
    nred = length(y) >> N
    skey = _sorted_key(key)
    offs = _offsets(key)
    Uc = SMatrix{D,D,OT}(U)
    if offs == ntuple(j -> j - 1, Val(D))
        K = min(_KERNEL_BATCH[], D)
        nchunk, ntail = divrem(nred, K)
        nchunk > 0 && _run_range(nchunk) do a, b
            _lmul_batched_range!(y, x, Uc, a, b, Val(K), Val(N))
        end
        ntail > 0 && _lmul_single_range!(y, x, Uc, skey, offs, nchunk * K, nred - 1)
    else
        _run_range(nred) do a, b
            _lmul_single_range!(y, x, Uc, skey, offs, a, b)
        end
    end
    return y
end

"""
    axpy_kernel!(y::AbstractVector, α::Number, x::AbstractVector, key, U) -> y

`y += α·(U·x)`（就地累加，无中间数组；`y` 与 `x` 不得共享存储）。
"""
function axpy_kernel!(y::AbstractVector, α::Number, x::AbstractVector,
                      key::NTuple{N,Int}, U::AbstractMatrix) where {N}
    D = 1 << N
    (size(U, 1) == size(U, 2) == D) ||
        throw(ArgumentError("matrix size $(size(U)) does not match $N qubit(s)"))
    length(y) == length(x) || throw(DimensionMismatch("state length mismatch"))
    OT = promote_type(eltype(y), eltype(x))
    nred = length(y) >> N
    skey = _sorted_key(key)
    offs = _offsets(key)
    Uc = SMatrix{D,D,OT}(U)
    αc = convert(OT, α)
    if offs == ntuple(j -> j - 1, Val(D))
        K = min(_KERNEL_BATCH[], D)
        nchunk, ntail = divrem(nred, K)
        nchunk > 0 && _run_range(nchunk) do a, b
            _axpy_batched_range!(y, αc, x, Uc, a, b, Val(K), Val(N))
        end
        ntail > 0 && _axpy_single_range!(y, αc, x, Uc, skey, offs, nchunk * K, nred - 1)
    else
        _run_range(nred) do a, b
            _axpy_single_range!(y, αc, x, Uc, skey, offs, a, b)
        end
    end
    return y
end

function _lmul_batched_range!(y::AbstractVector, x::AbstractVector,
                              U::SMatrix{D,D,OT}, istart::Int, iend::Int,
                              ::Val{K}, ::Val{N}) where {OT,D,K,N}
    len = D * K
    @inbounds for t in istart:iend
        base0 = len * t
        V = SMatrix{D,K,OT}(ntuple(j -> x[base0 + j], Val(len)))
        O = U * V
        for j in 1:len
            y[base0 + j] = O[j]
        end
    end
    return nothing
end

function _lmul_single_range!(y::AbstractVector, x::AbstractVector,
                             U::SMatrix{D,D,OT}, skey::NTuple{N,Int},
                             offs::NTuple{D,Int}, istart::Int, iend::Int) where {OT,D,N}
    @inbounds for rest in istart:iend
        base = _bit_insert_zeros(rest, skey)
        loc = SVector{D,OT}(ntuple(i -> x[base + offs[i] + 1], Val(D)))
        out = U * loc
        for i in 1:D
            y[base + offs[i] + 1] = out[i]
        end
    end
    return nothing
end

function _axpy_batched_range!(y::AbstractVector{T}, α::OT, x::AbstractVector{T},
                              U::SMatrix{D,D,OT}, istart::Int, iend::Int,
                              ::Val{K}, ::Val{N}) where {T,OT,D,K,N}
    len = D * K
    @inbounds for t in istart:iend
        base0 = len * t
        V = SMatrix{D,K,T}(ntuple(j -> x[base0 + j], Val(len)))
        O = U * V
        for j in 1:len
            y[base0 + j] += α * O[j]
        end
    end
    return nothing
end

function _axpy_single_range!(y::AbstractVector{T}, α::OT, x::AbstractVector{T},
                             U::SMatrix{D,D,OT}, skey::NTuple{N,Int},
                             offs::NTuple{D,Int}, istart::Int, iend::Int) where {T,OT,D,N}
    @inbounds for rest in istart:iend
        base = _bit_insert_zeros(rest, skey)
        loc = SVector{D,T}(ntuple(i -> x[base + offs[i] + 1], Val(D)))
        out = U * loc
        for i in 1:D
            y[base + offs[i] + 1] += α * out[i]
        end
    end
    return nothing
end

# DM 平坦存储（列 major，`ρ[i + d·j]`）上的左乘：只作用行索引（列不变），
# 行局域块 gather 跨步为 `d`。

function dm_lmul_kernel!(y::AbstractVector, x::AbstractVector, d::Int,
                         key::NTuple{N,Int}, U::AbstractMatrix) where {N}
    D = 1 << N
    (size(U, 1) == size(U, 2) == D) ||
        throw(ArgumentError("matrix size $(size(U)) does not match $N qubit(s)"))
    length(y) == length(x) == d * d || throw(DimensionMismatch("density storage mismatch"))
    OT = promote_type(eltype(y), eltype(x))
    nred = d >> N
    skey = _sorted_key(key)
    offs = _offsets(key)
    Uc = SMatrix{D,D,OT}(U)
    _run_range(nred * d) do a, b
        _dm_lmul_single_range!(y, x, d, Uc, skey, offs, a, b)
    end
    return y
end

function dm_axpy_kernel!(y::AbstractVector, α::Number, x::AbstractVector, d::Int,
                         key::NTuple{N,Int}, U::AbstractMatrix) where {N}
    D = 1 << N
    (size(U, 1) == size(U, 2) == D) ||
        throw(ArgumentError("matrix size $(size(U)) does not match $N qubit(s)"))
    length(y) == length(x) == d * d || throw(DimensionMismatch("density storage mismatch"))
    OT = promote_type(eltype(y), eltype(x))
    nred = d >> N
    skey = _sorted_key(key)
    offs = _offsets(key)
    Uc = SMatrix{D,D,OT}(U)
    αc = convert(OT, α)
    _run_range(nred * d) do a, b
        _dm_axpy_single_range!(y, αc, x, d, Uc, skey, offs, a, b)
    end
    return y
end

function _dm_lmul_single_range!(y::AbstractVector, x::AbstractVector, d::Int,
                                U::SMatrix{D,D,OT}, skey::NTuple{N,Int},
                                offs::NTuple{D,Int}, istart::Int, iend::Int) where {OT,D,N}
    @inbounds for rest in istart:iend
        rr, c = divrem(rest, d)          # 行非 key 位组合 / 完整列（左乘不改列）
        base = _bit_insert_zeros(rr, skey)
        loc = SVector{D,OT}(ntuple(li -> x[base + offs[li] + d * c + 1], Val(D)))
        out = U * loc
        for li in 1:D
            y[base + offs[li] + d * c + 1] = out[li]
        end
    end
    return nothing
end

function _dm_axpy_single_range!(y::AbstractVector{T}, α::OT, x::AbstractVector, d::Int,
                                U::SMatrix{D,D,OT}, skey::NTuple{N,Int},
                                offs::NTuple{D,Int}, istart::Int, iend::Int) where {T,OT,D,N}
    @inbounds for rest in istart:iend
        rr, c = divrem(rest, d)
        base = _bit_insert_zeros(rr, skey)
        loc = SVector{D,OT}(ntuple(li -> x[base + offs[li] + d * c + 1], Val(D)))
        out = U * loc
        for li in 1:D
            y[base + offs[li] + d * c + 1] += α * out[li]
        end
    end
    return nothing
end

# ── multi-factor expect：多因子局域算符链的期望值 ─────────────────────────────

"""
    multi_expect_kernel(v::AbstractVector, factors) -> scalar

计算多因子局域算符链的期望值 `⟨v|U₁U₂…Uₖ|v⟩`——**不复制量子态、
无 `2ⁿ` 中间数组、不构造多因子大矩阵**。`factors` 为 `(key, U)`
元组向量（`key` LSB-first、0-based），按矩阵积从右到左排列；
各因子的比特可以不同。

实现：取所有因子比特的并集做**联合分块**（块大小 `2^K`，`K` 为
并集位数），逐块 gather 后在块缓冲内从右到左逐因子就地作用，最后
`dot` 累加。函数内唯一的大对象是一个 `2^K` 块缓冲——与系统大小
`n` 无关，只随因子并集位数指数增长（典型 Pauli 串 `K ≤ 6`）。
"""
function multi_expect_kernel(v::AbstractVector{T}, factors) where {T}
    isempty(factors) && throw(ArgumentError("empty factor list"))
    # 联合比特并集（升序）
    allq = Int[]
    for (key, _) in factors, q in key
        q in allq || push!(allq, q)
    end
    sort!(allq)
    K = length(allq)
    nblocks = length(v) >> K
    nblocks >= 1 || throw(ArgumentError("factor qubit union too large for system size"))
    OT = T
    for (_, U) in factors
        OT = promote_type(OT, eltype(U))
    end
    skey = NTuple{K,Int}(allq)
    L = 1 << K
    # 联合块内局域地址：把局域索引 j 的位散射到 allq 位（base 的 allq 位
    # 为 0，相加无进位）。预计算一次。
    addr = [1 + _scatter(j, skey) for j in 0:L-1]
    return _run_range_sum(OT, nblocks) do a, b
        buf = Vector{OT}(undef, L)         # 每线程独立，避免竞争
        acc_ = zero(OT)
        @inbounds for rest in a:b
            base = _bit_insert_zeros(rest, skey)
            # gather 联合块
            for i in 1:L
                buf[i] = v[base + addr[i]]
            end
            # 从右到左逐因子在块内就地作用
            for (fkey, U) in factors
                subkey = ntuple(i -> searchsortedfirst(allq, fkey[i]) - 1, length(fkey))
                apply_kernel!(buf, subkey, U)
            end
            # ⟨v_block | result_block⟩
            s = zero(OT)
            for i in 1:L
                s += conj(v[base + addr[i]]) * buf[i]
            end
            acc_ += s
        end
        return acc_
    end
end

"""
    dm_multi_expect_kernel(rho::AbstractVector, d::Int, factors, ws::AbstractVector) -> scalar

密度矩阵版本：`tr(ρ·U₁U₂…Uₖ)`（`rho` 为长度 `d²` 的列 major 平坦存储，
左乘链只作用行索引）。内存特性与 `multi_expect_kernel` 相同。
"""
function dm_multi_expect_kernel(rho::AbstractVector, d::Int, factors, ws::AbstractVector)
    length(ws) == length(rho) == d * d || throw(DimensionMismatch("density storage mismatch"))
    ws === rho && throw(ArgumentError("workspace must not alias the state"))
    key, U = factors[1]
    dm_lmul_kernel!(ws, rho, d, key, U)
    for i in 2:length(factors)
        apply_kernel!(ws, factors[i]...)   # 就地行作用
    end
    return sum(ws[i * d + i + 1] for i in 0:d-1)   # tr(ws) = tr(ρ·U₁…Uₖ)
end

# ── expect：局域矩阵元 ⟨v|U|v⟩ / ⟨vc|U|v⟩ ───────────────────────────────────

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
    offs = _offsets(key)
    Uc = SMatrix{D,D,OT}(U)
    acc = zero(OT)
    if offs == ntuple(j -> j - 1, Val(D))
        K = min(_KERNEL_BATCH[], D)
        nchunk, ntail = divrem(nred, K)
        if nchunk > 0
            acc += _run_range_sum(OT, nchunk) do a, b
                _expect_batched_range(v, Uc, a, b, Val(K), Val(N))
            end
        end
        if ntail > 0
            acc += _expect_single_range(v, Uc, skey, offs, nchunk * K, nred - 1)
        end
    else
        acc = _run_range_sum(OT, nred) do a, b
            _expect_single_range(v, Uc, skey, offs, a, b)
        end
    end
    return acc
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
    offs = _offsets(key)
    Uc = SMatrix{D,D,OT}(U)
    acc = zero(OT)
    if offs == ntuple(j -> j - 1, Val(D))
        K = min(_KERNEL_BATCH[], D)
        nchunk, ntail = divrem(nred, K)
        if nchunk > 0
            acc += _run_range_sum(OT, nchunk) do a, b
                _expect2_batched_range(vc, v, Uc, a, b, Val(K), Val(N))
            end
        end
        if ntail > 0
            acc += _expect2_single_range(vc, v, Uc, skey, offs, nchunk * K, nred - 1)
        end
    else
        acc = _run_range_sum(OT, nred) do a, b
            _expect2_single_range(vc, v, Uc, skey, offs, a, b)
        end
    end
    return acc
end

"`K` 个连续块的 `Σₖ ⟨vₖ|U|vₖ⟩ = sum(conj(V) .* (U * V))` 批量累积。"
function _expect_batched_range(v::AbstractVector{T}, U::SMatrix{D,D,OT},
                               istart::Int, iend::Int, ::Val{K}, ::Val{N}) where {T,OT,D,K,N}
    len = D * K
    acc = zero(OT)
    @inbounds for t in istart:iend
        base0 = len * t
        V = SMatrix{D,K,T}(ntuple(j -> v[base0 + j], Val(len)))
        acc += sum(conj(V) .* (U * V))
    end
    return acc
end

"`K` 个连续块的 `Σₖ ⟨vcₖ|U|vₖ⟩` 批量累积。"
function _expect2_batched_range(vc::AbstractVector, v::AbstractVector{T},
                                U::SMatrix{D,D,OT}, istart::Int, iend::Int,
                                ::Val{K}, ::Val{N}) where {T,OT,D,K,N}
    len = D * K
    acc = zero(OT)
    @inbounds for t in istart:iend
        base0 = len * t
        V = SMatrix{D,K,T}(ntuple(j -> v[base0 + j], Val(len)))
        Vc = SMatrix{D,K,OT}(ntuple(j -> vc[base0 + j], Val(len)))
        acc += sum(conj(Vc) .* (U * V))
    end
    return acc
end

function _expect_single_range(v::AbstractVector{T}, U::SMatrix{D,D,OT},
                              skey::NTuple{N,Int}, offs::NTuple{D,Int},
                              istart::Int, iend::Int) where {T,OT,D,N}
    acc = zero(OT)
    @inbounds for rest in istart:iend
        base = _bit_insert_zeros(rest, skey)
        loc = SVector{D,T}(ntuple(i -> v[base + offs[i] + 1], Val(D)))
        acc += dot(loc, U, loc)
    end
    return acc
end

function _expect2_single_range(vc::AbstractVector, v::AbstractVector{T},
                               U::SMatrix{D,D,OT}, skey::NTuple{N,Int},
                               offs::NTuple{D,Int}, istart::Int, iend::Int) where {T,OT,D,N}
    acc = zero(OT)
    @inbounds for rest in istart:iend
        base = _bit_insert_zeros(rest, skey)
        loc = SVector{D,T}(ntuple(i -> v[base + offs[i] + 1], Val(D)))
        loc_c = SVector{D,eltype(vc)}(ntuple(i -> vc[base + offs[i] + 1], Val(D)))
        acc += dot(loc_c, U, loc)
    end
    return acc
end

# ── dm_expect：局域矩阵元 tr(ρ m) ────────────────────────────────────────────

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
    offs = _offsets(key)
    mvs = SVector{D*D,OT}(mv)
    acc = zero(OT)
    if offs == ntuple(j -> j - 1, Val(D))
        K = min(_KERNEL_BATCH[], D)
        nchunk, ntail = divrem(nred, K)
        if nchunk > 0
            acc += _run_range_sum(OT, nchunk) do a, b
                _dm_expect_batched_range(rho, d, mvs, skey, offs, a, b, Val(K), Val(N))
            end
        end
        if ntail > 0
            acc += _dm_expect_single_range(rho, d, mvs, skey, offs, nchunk * K, nred - 1)
        end
    else
        acc = _run_range_sum(OT, nred) do a, b
            _dm_expect_single_range(rho, d, mvs, skey, offs, a, b)
        end
    end
    return acc
end

"`K` 个连续块的 `Σₖ tr(ρₖ m)` 批量累积（局域块收集跨步 `d`）。"
function _dm_expect_batched_range(rho::AbstractVector{T}, d::Int, mv::SVector{D2,OT},
                                  skey::NTuple{N,Int}, offs::NTuple{D,Int},
                                  istart::Int, iend::Int, ::Val{K}, ::Val{N}) where {T,OT,D,N,D2,K}
    kk = trailing_zeros(K)
    skey_pfx = ntuple(i -> skey[i], Val(kk))
    Pk = ntuple(k -> _bit_insert_zeros(k - 1, skey), Val(K))
    Mv = hcat(fill(mv, K)...)   # 每列都是 mv
    acc = zero(OT)
    @inbounds for t in istart:iend
        base0 = _bit_insert_zeros(K * t, skey)
        V = SMatrix{D2,K,T}(ntuple(j -> begin
            k, i = fldmod(j - 1, D2)
            lj, li = divrem(i, D)        # i = li + D*lj（列 major：li 为行局域索引）
            b = base0 + Pk[k+1]
            rho[b + li + d * (b + lj) + 1]
        end, Val(D2 * K)))
        acc += sum(V .* Mv)
    end
    return acc
end

function _dm_expect_single_range(rho::AbstractVector{T}, d::Int, mv::SVector{D2,OT},
                                 skey::NTuple{N,Int}, offs::NTuple{D,Int},
                                 istart::Int, iend::Int) where {T,OT,D,N,D2}
    acc = zero(OT)
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
