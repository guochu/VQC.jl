
"""
    applys when key > 3, U is the transposed op
"""
function _expectation_value_util_H(key::Int, U::AbstractMatrix, v::AbstractVector)
    L = length(v)
    sizek = 1 << (key - 1)
    mask0 = sizek - 1
    mask1 = xor(L - 1, 2 * sizek - 1)
    f(ist::Int, ifn::Int, pos::Int, m1::Int, m2::Int, mat::AbstractMatrix, p::AbstractVector) = begin
    	r = zero(eltype(p))
        @inbounds for i in ist:ifn
            l = (16 * i & m2) | (8 * i & m1) + 1
            l1 = l + pos
            vi = SMatrix{8, 2}(p[l], p[l+1], p[l+2], p[l+3], p[l+4], p[l+5], p[l+6], p[l+7],
            p[l1], p[l1+1], p[l1+2], p[l1+3], p[l1+4], p[l1+5], p[l1+6], p[l1+7])

            vi_t = transpose(vi)
            @fastmath r += dot(vi_t, mat, vi_t)
        end
        return r
    end

	return parallel_sum(eltype(v), div(L, 16), Threads.nthreads(), f, sizek, mask0, mask1, U, v)
end

"""
    applys when key <= 3, U is the transposed op
"""
function _expectation_value_util_L(key::Int, U::AbstractMatrix, v::AbstractVector)
    L = length(v)
    f1(ist::Int, ifn::Int, mat::AbstractMatrix, p::AbstractVector) = begin
    	r = zero(eltype(p))
        @inbounds for i in ist:ifn
            l = 16 * i + 1
            vi = SMatrix{2, 8}(p[l], p[l+1], p[l+2], p[l+3], p[l+4], p[l+5], p[l+6], p[l+7],
            p[l+8], p[l+9], p[l+10], p[l+11], p[l+12], p[l+13], p[l+14], p[l+15])

            @fastmath r += dot(vi, mat, vi)
        end
        return r
    end
    f2(ist::Int, ifn::Int, mat::AbstractMatrix, p::AbstractVector) = begin
    	r = zero(eltype(p))
        @inbounds for i in ist:ifn
            l = 16 * i + 1
            vi = SMatrix{2, 8}(p[l], p[l+2], p[l+1], p[l+3], p[l+4], p[l+6], p[l+5], p[l+7],
            p[l+8], p[l+10], p[l+9], p[l+11], p[l+12], p[l+14], p[l+13], p[l+15])

            @fastmath r += dot(vi, mat, vi)
        end
        return r
    end
    f3(ist::Int, ifn::Int, mat::AbstractMatrix, p::AbstractVector) = begin
    	r = zero(eltype(p))
        @inbounds for i in ist:ifn
            l = 16 * i + 1
            vi = SMatrix{2, 8}(p[l], p[l+4], p[l+1], p[l+5], p[l+2], p[l+6], p[l+3], p[l+7],
            p[l+8], p[l+12], p[l+9], p[l+13], p[l+10], p[l+14], p[l+11], p[l+15])

            @fastmath r += dot(vi, mat, vi)
        end
        return r
    end
    if key == 1
        f = f1
    elseif key == 2
        f = f2
    elseif key == 3
        f = f3
    else
        error("qubit position $key not allowed for L.")
    end

    parallel_sum(eltype(v), div(L, 16), Threads.nthreads(), f, U, v)
end

function _expectation_value_threaded_util(q0::Int, U::AbstractMatrix, v::AbstractVector)
    if q0 > 3
        return _expectation_value_util_H(q0, SMatrix{2,2, eltype(v)}(U), v)
    else
        return _expectation_value_util_L(q0, SMatrix{2,2, eltype(v)}(U), v)
    end
end
_expectation_value_threaded_util(q0::Tuple{Int}, U::AbstractMatrix, v::AbstractVector) = _expectation_value_threaded_util(
    q0[1], U, v)



"""
    applys when both keys > 3, U is the transposed op
"""
function _expectation_value_util_HH(key::Tuple{Int, Int}, U::AbstractMatrix, v::AbstractVector)
    L = length(v)
    q1, q2 = key
    pos1, pos2 = 1 << (q1-1), 1 << (q2-1)
    # stride2, stride3 = pos1 << 1, pos2 << 1
    mask0 = pos1 - 1
    mask1 = xor(pos2 - 1, 2 * pos1 - 1)
    mask2 = xor(L - 1, 2 * pos2 - 1)
    # println("pos1=$pos1, pos2=$pos2, m0=$mask0, m1=$mask1, m2=$mask2")

    total_itr = div(L, 32)
    f(ist::Int, ifn::Int, posa::Int, posb::Int, m1::Int, m2::Int, m3::Int, mat::AbstractMatrix, p::AbstractVector) = begin
    	r = zero(eltype(p))
        @inbounds for i in ist:ifn
            l = (32 * i & m3) | (16 * i & m2) | (8 * i & m1) + 1
            # l = div(l, 2) + 1
            l1 = l + posa
            l2 = l + posb
            l3 = l2 + posa
            # println("l0=$l, l1=$l1, l2=$l2, l3=$l3")
            vi = SMatrix{8, 4}(p[l], p[l+1], p[l+2], p[l+3], p[l+4], p[l+5], p[l+6], p[l+7],
            p[l1], p[l1+1], p[l1+2], p[l1+3], p[l1+4], p[l1+5], p[l1+6], p[l1+7],
            p[l2], p[l2+1], p[l2+2], p[l2+3], p[l2+4], p[l2+5], p[l2+6], p[l2+7],
            p[l3], p[l3+1], p[l3+2], p[l3+3], p[l3+4], p[l3+5], p[l3+6], p[l3+7])

            vi_t = transpose(vi)
            @fastmath r += dot(vi_t, mat, vi_t)
        end
        return r
    end

    parallel_sum(eltype(v), total_itr, Threads.nthreads(), f, pos1, pos2, mask0, mask1, mask2, U, v)
end


"""
    applys when q1 <= 3 and q2 > 4, U is the transposed op
"""
function _expectation_value_util_LH(key::Tuple{Int, Int}, U::AbstractMatrix, v::AbstractVector)
    L = length(v)
    q1, q2 = key
    sizej = 1 << (q2-1)
    mask0 = sizej - 1
    mask1 = xor(L - 1, 2 * sizej - 1)

    f1H(ist::Int, ifn::Int, posb::Int, m1::Int, m2::Int, mat::AbstractMatrix, p::AbstractVector) = begin
    	r = zero(eltype(p))
        @inbounds for i in ist:ifn
            l = (32 * i & m2) | (16 * i & m1)
            l1 = l + posb
            # println("l=$l, l1=$l1")
            vi = SMatrix{8, 4}(p[l+1], p[l+3], p[l+5], p[l+7], p[l+9], p[l+11], p[l+13], p[l+15],
            p[l+2], p[l+4], p[l+6], p[l+8], p[l+10], p[l+12], p[l+14], p[l+16],
            p[l1+1], p[l1+3], p[l1+5], p[l1+7], p[l1+9], p[l1+11], p[l1+13], p[l1+15],
            p[l1+2], p[l1+4], p[l1+6], p[l1+8], p[l1+10], p[l1+12], p[l1+14], p[l1+16])

            vi_t = transpose(vi)
            @fastmath r += dot(vi_t, mat, vi_t)
        end
        return r
    end

    f2H(ist::Int, ifn::Int, posb::Int, m1::Int, m2::Int, mat::AbstractMatrix, p::AbstractVector) = begin
    	r = zero(eltype(p))
        @inbounds for i in ist:ifn
            l = (32 * i & m2) | (16 * i & m1)
            l1 = l + posb
            # println("l=$l, l1=$l1")
            vi = SMatrix{8, 4}(p[l+1], p[l+2], p[l+5], p[l+6], p[l+9], p[l+10], p[l+13], p[l+14],
            p[l+3], p[l+4], p[l+7], p[l+8], p[l+11], p[l+12], p[l+15], p[l+16],
            p[l1+1], p[l1+2], p[l1+5], p[l1+6], p[l1+9], p[l1+10], p[l1+13], p[l1+14],
            p[l1+3], p[l1+4], p[l1+7], p[l1+8], p[l1+11], p[l1+12], p[l1+15], p[l1+16])

            vi_t = transpose(vi)
            @fastmath r += dot(vi_t, mat, vi_t)
        end
        return r
    end

    f3H(ist::Int, ifn::Int, posb::Int, m1::Int, m2::Int, mat::AbstractMatrix, p::AbstractVector) = begin
    	r = zero(eltype(p))
        @inbounds for i in ist:ifn
            l = (32 * i & m2) | (16 * i & m1)
            l1 = l + posb
            # println("l=$l, l1=$l1")
            vi = SMatrix{8, 4}(p[l+1], p[l+2], p[l+3], p[l+4], p[l+9], p[l+10], p[l+11], p[l+12],
            p[l+5], p[l+6], p[l+7], p[l+8], p[l+13], p[l+14], p[l+15], p[l+16],
            p[l1+1], p[l1+2], p[l1+3], p[l1+4], p[l1+9], p[l1+10], p[l1+11], p[l1+12],
            p[l1+5], p[l1+6], p[l1+7], p[l1+8], p[l1+13], p[l1+14], p[l1+15], p[l1+16])

            vi_t = transpose(vi)
            @fastmath r += dot(vi_t, mat, vi_t)
        end
        return r
    end

    if q1 == 1
        f = f1H
    elseif q1 == 2
        f = f2H
    elseif q1 == 3
        f = f3H
    else
        error("qubit position $q1 not allowed for LH.")
    end
    total_itr = div(L, 32)

    parallel_sum(eltype(v), total_itr, Threads.nthreads(), f, sizej, mask0, mask1, U, v)
end

"""
    applys when both keys <= 4
"""
function _expectation_value_util_LL(key::Tuple{Int, Int}, U::AbstractMatrix, v::AbstractVector)
    L = length(v)
    q1, q2 = key
    f12(ist::Int, ifn::Int, mat::AbstractMatrix, p::AbstractVector) = begin
    	r = zero(eltype(p))
        @inbounds for i in ist:ifn
            l = 32 * i + 1
            vi = SMatrix{4, 8}(p[l], p[l+1], p[l+2], p[l+3], p[l+4], p[l+5], p[l+6], p[l+7],
            p[l+8], p[l+9], p[l+10], p[l+11], p[l+12], p[l+13], p[l+14], p[l+15],
            p[l+16], p[l+17], p[l+18], p[l+19], p[l+20], p[l+21], p[l+22], p[l+23],
            p[l+24], p[l+25], p[l+26], p[l+27], p[l+28], p[l+29], p[l+30], p[l+31])

            @fastmath r += dot(vi, mat, vi)
        end
        return r
    end
    f13(ist::Int, ifn::Int, mat::AbstractMatrix, p::AbstractVector) = begin
    	r = zero(eltype(p))
        @inbounds for i in ist:ifn
            l = 32 * i + 1
            vi = SMatrix{4, 8}(p[l], p[l+1], p[l+4], p[l+5], p[l+2], p[l+3], p[l+6], p[l+7],
            p[l+8], p[l+9], p[l+12], p[l+13], p[l+10], p[l+11], p[l+14], p[l+15],
            p[l+16], p[l+17], p[l+20], p[l+21], p[l+18], p[l+19], p[l+22], p[l+23],
            p[l+24], p[l+25], p[l+28], p[l+29], p[l+26], p[l+27], p[l+30], p[l+31])

            @fastmath r += dot(vi, mat, vi)
        end
        return r
    end
    f14(ist::Int, ifn::Int, mat::AbstractMatrix, p::AbstractVector) = begin
    	r = zero(eltype(p))
        @inbounds for i in ist:ifn
            l = 32 * i
            vi = SMatrix{4, 8}(p[l+1], p[l+2], p[l+9], p[l+10], p[l+3], p[l+4], p[l+11], p[l+12],
            p[l+5], p[l+6], p[l+13], p[l+14], p[l+7], p[l+8], p[l+15], p[l+16],
            p[l+17], p[l+18], p[l+25], p[l+26], p[l+19], p[l+20], p[l+27], p[l+28],
            p[l+21], p[l+22], p[l+29], p[l+30], p[l+23], p[l+24], p[l+31], p[l+32])

            @fastmath r += dot(vi, mat, vi)
        end
        return r
    end
    f23(ist::Int, ifn::Int, mat::AbstractMatrix, p::AbstractVector) = begin
   	 	r = zero(eltype(p))
        @inbounds for i in ist:ifn
            l = 32 * i + 1
            vi = SMatrix{4, 8}(p[l], p[l+2], p[l+4], p[l+6], p[l+1], p[l+3], p[l+5], p[l+7],
            p[l+8], p[l+10], p[l+12], p[l+14], p[l+9], p[l+11], p[l+13], p[l+15],
            p[l+16], p[l+18], p[l+20], p[l+22], p[l+17], p[l+19], p[l+21], p[l+23],
            p[l+24], p[l+26], p[l+28], p[l+30], p[l+25], p[l+27], p[l+29], p[l+31])

            @fastmath r += dot(vi, mat, vi)
        end
        return r
    end
    f24(ist::Int, ifn::Int, mat::AbstractMatrix, p::AbstractVector) = begin
    	r = zero(eltype(p))
        @inbounds for i in ist:ifn
            l = 32 * i
            vi = SMatrix{4, 8}(p[l+1], p[l+3], p[l+9], p[l+11], p[l+2], p[l+4], p[l+10], p[l+12],
            p[l+5], p[l+7], p[l+13], p[l+15], p[l+6], p[l+8], p[l+14], p[l+16],
            p[l+17], p[l+19], p[l+25], p[l+27], p[l+18], p[l+20], p[l+26], p[l+28],
            p[l+21], p[l+23], p[l+29], p[l+31], p[l+22], p[l+24], p[l+30], p[l+32])

            @fastmath r += dot(vi, mat, vi)
        end
        return r
    end
    f34(ist::Int, ifn::Int, mat::AbstractMatrix, p::AbstractVector) = begin
    	r = zero(eltype(p))
        @inbounds for i in ist:ifn
            l = 32 * i
            vi = SMatrix{4, 8}(p[l+1], p[l+5], p[l+9], p[l+13], p[l+2], p[l+6], p[l+10], p[l+14],
            p[l+3], p[l+7], p[l+11], p[l+15], p[l+4], p[l+8], p[l+12], p[l+16],
            p[l+17], p[l+21], p[l+25], p[l+29], p[l+18], p[l+22], p[l+26], p[l+30],
            p[l+19], p[l+23], p[l+27], p[l+31], p[l+20], p[l+24], p[l+28], p[l+32])

            @fastmath r += dot(vi, mat, vi)
        end
        return r
    end
    if q1==1 && q2 == 2
        f = f12
    elseif q1==1 && q2 == 3
        f = f13
    elseif q1==1 && q2 == 4
        f = f14
    elseif q1==2 && q2 == 3
        f = f23
    elseif q1==2 && q2 == 4
        f = f24
    elseif q1==3 && q2 == 4
        f = f34
    else
        error("qubit position $q1 and $q2 not allowed for LL.")
    end
    total_itr = div(L, 32)

    parallel_sum(eltype(v), total_itr, Threads.nthreads(), f, U, v)
end


function _expectation_value_threaded_util(key::Tuple{Int, Int}, U::AbstractMatrix, v::AbstractVector)
    q0, q1 = key
    if q0 > 3
        return _expectation_value_util_HH(key, SMatrix{4,4, eltype(v)}(U), v)
    elseif q1 > 4
        return _expectation_value_util_LH(key, SMatrix{4,4, eltype(v)}(U), v)
    else
        return _expectation_value_util_LL(key, SMatrix{4,4, eltype(v)}(U), v)
    end
end


"""
    applys when both keys > 2, U is assumed to be transposed
"""
function _expectation_value_util_HHH(key::Tuple{Int, Int, Int}, U::AbstractMatrix, v::AbstractVector)
    L = length(v)
    q1, q2, q3 = key
    pos1, pos2, pos3 = 1 << (q1-1), 1 << (q2-1), 1 << (q3-1)
    # stride2, stride3 = pos1 << 1, pos2 << 1
    mask0 = pos1 - 1
    mask1 = xor(pos2 - 1, 2 * pos1 - 1)
    mask2 = xor(pos3 - 1, 2 * pos2 - 1)
    mask3 = xor(L - 1, 2 * pos3 - 1)
    # println("pos1=$pos1, pos2=$pos2, m0=$mask0, m1=$mask1, m2=$mask2")

    total_itr = div(L, 32)
    f(ist::Int, ifn::Int, posa::Int, posb::Int, posc::Int, m1::Int, m2::Int, m3::Int, m4::Int, mat::AbstractMatrix, p::AbstractVector) = begin
    	r = zero(eltype(p))
        @inbounds for i in ist:ifn
            l000 = (32 * i & m4) | (16 * i & m3) | (8 * i & m2) | (4 * i & m1) + 1
            l100 = l000 + posa
            l010 = l000 + posb
            l110 = l010 + posa

            l001 = l000 + posc
            l101 = l001 + posa
            l011 = l001 + posb
            l111 = l011 + posa

            # println("$l000, $l100, $l010, $l110, $l001, $l101, $l011, $l111")
            vi = SMatrix{4, 8}(p[l000], p[l000+1], p[l000+2], p[l000+3],
                               p[l100], p[l100+1], p[l100+2], p[l100+3],
                               p[l010], p[l010+1], p[l010+2], p[l010+3],
                               p[l110], p[l110+1], p[l110+2], p[l110+3],
                               p[l001], p[l001+1], p[l001+2], p[l001+3],
                               p[l101], p[l101+1], p[l101+2], p[l101+3],
                               p[l011], p[l011+1], p[l011+2], p[l011+3],
                               p[l111], p[l111+1], p[l111+2], p[l111+3])

            vi_t = transpose(vi)

            @fastmath r += dot(vi_t, mat, vi_t)
        end
        return r
    end

    parallel_sum(eltype(v), total_itr, Threads.nthreads(), f, pos1, pos2, pos3, mask0, mask1, mask2, mask3, U, v)
end


"""
    applys when q1 <= 2 and q2 > 3, U is the transposed op
"""
function _expectation_value_util_LHH(key::Tuple{Int, Int, Int}, U::AbstractMatrix, v::AbstractVector)
    L = length(v)
    q1, q2, q3 = key
    sizej, sizel = 1 << (q2-1), 1 << (q3-1)
    mask0 = sizej - 1
    mask1 = xor(sizel - 1, 2 * sizej - 1)
    mask2 = xor(L-1, 2 * sizel - 1)

    f1H(ist::Int, ifn::Int, posb::Int, posc::Int, m1::Int, m2::Int, m3::Int, mat::AbstractMatrix, p::AbstractVector) = begin
    	r = zero(eltype(p))
        @inbounds for i in ist:ifn
            l = (32 * i & m3) | (16 * i & m2) | (8 * i & m1)
            l1 = l + posb
            l2 = l + posc
            l3 = l2 + posb
            # println("$l, $l1, $l2, $l3")
            vi = SMatrix{4, 8}(p[l+1], p[l+3], p[l+5], p[l+7],
                               p[l+2], p[l+4], p[l+6], p[l+8],
                               p[l1+1], p[l1+3], p[l1+5], p[l1+7],
                               p[l1+2], p[l1+4], p[l1+6], p[l1+8],
                               p[l2+1], p[l2+3], p[l2+5], p[l2+7],
                               p[l2+2], p[l2+4], p[l2+6], p[l2+8],
                               p[l3+1], p[l3+3], p[l3+5], p[l3+7],
                               p[l3+2], p[l3+4], p[l3+6], p[l3+8])

            vi_t = transpose(vi)
            @fastmath r += dot(vi_t, mat, vi_t)
        end
        return r
    end

    f2H(ist::Int, ifn::Int, posb::Int, posc::Int, m1::Int, m2::Int, m3::Int, mat::AbstractMatrix, p::AbstractVector) = begin
    	r = zero(eltype(p))
        @inbounds for i in ist:ifn
            l = (32 * i & m3) | (16 * i & m2) | (8 * i & m1)
            l1 = l + posb
            l2 = l + posc
            l3 = l2 + posb
            vi = SMatrix{4, 8}(p[l+1], p[l+2], p[l+5], p[l+6],
                               p[l+3], p[l+4], p[l+7], p[l+8],
                               p[l1+1], p[l1+2], p[l1+5], p[l1+6],
                               p[l1+3], p[l1+4], p[l1+7], p[l1+8],
                               p[l2+1], p[l2+2], p[l2+5], p[l2+6],
                               p[l2+3], p[l2+4], p[l2+7], p[l2+8],
                               p[l3+1], p[l3+2], p[l3+5], p[l3+6],
                               p[l3+3], p[l3+4], p[l3+7], p[l3+8])

            vi_t = transpose(vi)
            @fastmath r += dot(vi_t, mat, vi_t)
        end
        return r
    end

    f3H(ist::Int, ifn::Int, posb::Int, posc::Int, m1::Int, m2::Int, m3::Int, mat::AbstractMatrix, p::AbstractVector) = begin
    	r = zero(eltype(p))
        @inbounds for i in ist:ifn
            l = (32 * i & m3) | (16 * i & m2) | (8 * i & m1)
            l1 = l + posb
            l2 = l + posc
            l3 = l2 + posb
            vi = SMatrix{4, 8}(p[l+1], p[l+2], p[l+3], p[l+4],
                               p[l+5], p[l+6], p[l+7], p[l+8],
                               p[l1+1], p[l1+2], p[l1+3], p[l1+4],
                               p[l1+5], p[l1+6], p[l1+7], p[l1+8],
                               p[l2+1], p[l2+2], p[l2+3], p[l2+4],
                               p[l2+5], p[l2+6], p[l2+7], p[l2+8],
                               p[l3+1], p[l3+2], p[l3+3], p[l3+4],
                               p[l3+5], p[l3+6], p[l3+7], p[l3+8])

            vi_t = transpose(vi)
            @fastmath r += dot(vi_t, mat, vi_t)
        end
        return r
    end

    if q1 == 1
        f = f1H
    elseif q1 == 2
        f = f2H
    elseif q1 == 3
        f = f3H
    else
        error("qubit position $q1 not allowed for LHH.")
    end
    (q2 > 3) || error("qubit 2 position $q2 not allowed for LHH.")
    total_itr = div(L, 32)

    parallel_sum(eltype(v), total_itr, Threads.nthreads(), f, sizej, sizel, mask0, mask1, mask2, U, v)
end

"""
    applys when q1, q2 <= 3 and q3 > 4, U is the transposed op
"""
function _expectation_value_util_LLH(key::Tuple{Int, Int, Int}, U::AbstractMatrix, v::AbstractVector)
    L = length(v)
    q1, q2, q3 = key
    sizej = 1 << (q3-1)
    mask0 = sizej - 1
    mask1 = xor(L - 1, 2 * sizej - 1)

    f12H(ist::Int, ifn::Int, posb::Int, m1::Int, m2::Int, mat::AbstractMatrix, p::AbstractVector) = begin
    	r = zero(eltype(p))
        @inbounds for i in ist:ifn
            l = (32 * i & m2) | (16 * i & m1)
            l1 = l + posb
            # println("l=$l, l1=$l1")
            vi = SMatrix{4, 8}(p[l+1], p[l+5], p[l+9], p[l+13],
                               p[l+2], p[l+6], p[l+10], p[l+14],
                               p[l+3], p[l+7], p[l+11], p[l+15],
                               p[l+4], p[l+8], p[l+12], p[l+16],
                               p[l1+1], p[l1+5], p[l1+9], p[l1+13],
                               p[l1+2], p[l1+6], p[l1+10], p[l1+14],
                               p[l1+3], p[l1+7], p[l1+11], p[l1+15],
                               p[l1+4], p[l1+8], p[l1+12], p[l1+16])

 			vi_t = transpose(vi)
 			@fastmath r += dot(vi_t, mat, vi_t)
        end
        return r
    end

    f13H(ist::Int, ifn::Int, posb::Int, m1::Int, m2::Int, mat::AbstractMatrix, p::AbstractVector) = begin
    	r = zero(eltype(p))
        @inbounds for i in ist:ifn
            l = (32 * i & m2) | (16 * i & m1)
            l1 = l + posb
            # println("l=$l, l1=$l1")
            vi = SMatrix{4, 8}(p[l+1], p[l+3], p[l+9], p[l+11],
                               p[l+2], p[l+4], p[l+10], p[l+12],
                               p[l+5], p[l+7], p[l+13], p[l+15],
                               p[l+6], p[l+8], p[l+14], p[l+16],
                               p[l1+1], p[l1+3], p[l1+9], p[l1+11],
                               p[l1+2], p[l1+4], p[l1+10], p[l1+12],
                               p[l1+5], p[l1+7], p[l1+13], p[l1+15],
                               p[l1+6], p[l1+8], p[l1+14], p[l1+16])

 			vi_t = transpose(vi)
 			@fastmath r += dot(vi_t, mat, vi_t)
        end
        return r
    end

    f23H(ist::Int, ifn::Int, posb::Int, m1::Int, m2::Int, mat::AbstractMatrix, p::AbstractVector) = begin
    	r = zero(eltype(p))
        @inbounds for i in ist:ifn
            l = (32 * i & m2) | (16 * i & m1)
            l1 = l + posb
            # println("l=$l, l1=$l1")
            vi = SMatrix{4, 8}(p[l+1], p[l+2], p[l+9], p[l+10],
                               p[l+3], p[l+4], p[l+11], p[l+12],
                               p[l+5], p[l+6], p[l+13], p[l+14],
                               p[l+7], p[l+8], p[l+15], p[l+16],
                               p[l1+1], p[l1+2], p[l1+9], p[l1+10],
                               p[l1+3], p[l1+4], p[l1+11], p[l1+12],
                               p[l1+5], p[l1+6], p[l1+13], p[l1+14],
                               p[l1+7], p[l1+8], p[l1+15], p[l1+16])

             vi_t = transpose(vi)
 			@fastmath r += dot(vi_t, mat, vi_t)
        end
        return r
    end

    if q1 == 1 && q2 == 2
        f = f12H
    elseif q1 == 1 && q2 == 3
        f = f13H
    elseif q1 == 2 && q2 == 3
        f = f23H
    else
        error("qubit position $q1 not allowed for LH.")
    end
    total_itr = div(L, 32)

    parallel_sum(eltype(v), total_itr, Threads.nthreads(), f, sizej, mask0, mask1, U, v)
end


"""
    applys when both keys <= 4
"""
function _expectation_value_util_LLL(key::Tuple{Int, Int, Int}, U::AbstractMatrix, v::AbstractVector)
    L = length(v)
    q1, q2, q3 = key
    f123(ist::Int, ifn::Int, mat::AbstractMatrix, p::AbstractVector) = begin
    	r = zero(eltype(p))
        @inbounds for i in ist:ifn
            l = 32 * i + 1
            vi = SMatrix{8, 4}(p[l], p[l+1], p[l+2], p[l+3], p[l+4], p[l+5], p[l+6], p[l+7],
            p[l+8], p[l+9], p[l+10], p[l+11], p[l+12], p[l+13], p[l+14], p[l+15],
            p[l+16], p[l+17], p[l+18], p[l+19], p[l+20], p[l+21], p[l+22], p[l+23],
            p[l+24], p[l+25], p[l+26], p[l+27], p[l+28], p[l+29], p[l+30], p[l+31])

 			@fastmath r += dot(vi, mat, vi)
        end
        return r
    end
    f124(ist::Int, ifn::Int, mat::AbstractMatrix, p::AbstractVector) = begin
    	r = zero(eltype(p))
        @inbounds for i in ist:ifn
            l = 32 * i
            vi = SMatrix{8, 4}(p[l+1], p[l+2], p[l+3], p[l+4], p[l+9], p[l+10], p[l+11], p[l+12],
                               p[l+5], p[l+6], p[l+7], p[l+8], p[l+13], p[l+14], p[l+15], p[l+16],
                               p[l+17], p[l+18], p[l+19], p[l+20], p[l+25], p[l+26], p[l+27], p[l+28],
                               p[l+21], p[l+22], p[l+23], p[l+24], p[l+29], p[l+30], p[l+31], p[l+32])

            @fastmath r += dot(vi, mat, vi)
        end
        return r
    end
    f134(ist::Int, ifn::Int, mat::AbstractMatrix, p::AbstractVector) = begin
    	r = zero(eltype(p))
        @inbounds for i in ist:ifn
            l = 32 * i
            vi = SMatrix{8, 4}(p[l+1], p[l+2], p[l+5], p[l+6], p[l+9], p[l+10], p[l+13], p[l+14],
                               p[l+3], p[l+4], p[l+7], p[l+8], p[l+11], p[l+12], p[l+15], p[l+16],
                               p[l+17], p[l+18], p[l+21], p[l+22], p[l+25], p[l+26], p[l+29], p[l+30],
                               p[l+19], p[l+20], p[l+23], p[l+24], p[l+27], p[l+28], p[l+31], p[l+32])

            @fastmath r += dot(vi, mat, vi)
        end
        return r
    end
    f234(ist::Int, ifn::Int, mat::AbstractMatrix, p::AbstractVector) = begin
    	r = zero(eltype(p))
        @inbounds for i in ist:ifn
            l = 32 * i
            vi = SMatrix{8, 4}(p[l+1], p[l+3], p[l+5], p[l+7], p[l+9], p[l+11], p[l+13], p[l+15],
                               p[l+2], p[l+4], p[l+6], p[l+8], p[l+10], p[l+12], p[l+14], p[l+16],
                               p[l+17], p[l+19], p[l+21], p[l+23], p[l+25], p[l+27], p[l+29], p[l+31],
                               p[l+18], p[l+20], p[l+22], p[l+24], p[l+26], p[l+28], p[l+30], p[l+32])

            @fastmath r += dot(vi, mat, vi)
        end
        return r
    end
    if q1==1 && q2 == 2 && q3 == 3
        f = f123
    elseif q1==1 && q2 == 2 && q3 == 4
        f = f124
    elseif q1==1 && q2 == 3 && q3 == 4
        f = f134
    elseif q1==2 && q2 == 3 && q3 == 4
        f = f234
    else
        error("qubit position $q1 and $q2 not allowed for LL.")
    end
    total_itr = div(L, 32)

    parallel_sum(eltype(v), total_itr, Threads.nthreads(), f, U, v)
end

function _expectation_value_threaded_util(key::Tuple{Int, Int, Int}, U::AbstractMatrix, v::AbstractVector)
    q0, q1, q2 = key
    if q0 > 2
        return _expectation_value_util_HHH(key, SMatrix{8,8, eltype(v)}(U), v)
    elseif q1 > 3
        return _expectation_value_util_LHH(key, SMatrix{8,8, eltype(v)}(U), v)
    elseif q2 > 4
        return _expectation_value_util_LLH(key, SMatrix{8,8, eltype(v)}(U), v)
    else
        return _expectation_value_util_LLL(key, SMatrix{8,8, eltype(v)}(U), v)
    end
end


expectation_value_threaded(pos::Int, m::AbstractMatrix, state::AbstractVector) = _expectation_value_threaded_util(
	pos, m, state)

expectation_value_threaded(pos::Tuple{Int}, m::AbstractMatrix, state::AbstractVector) = expectation_value_threaded(
	pos[1], m, state)

expectation_value_threaded(pos::Tuple{Int, Int}, m::AbstractMatrix, state::AbstractVector) = _expectation_value_threaded_util(
	pos, m, state)

expectation_value_threaded(pos::Tuple{Int, Int, Int}, m::AbstractMatrix, state::AbstractVector) = _expectation_value_threaded_util(
	pos, m, state)





