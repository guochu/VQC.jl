

function apply_threaded!(gt::SWAPGate, v::AbstractVector)
    (length(v) < 32) && return apply_serial!(gt, v)
    L = length(v)
    q1, q2 = ordered_positions(gt)
    pos1, pos2 = 1 << (q1-1), 1 << (q2-1)

    mask0 = pos1 - 1
    mask1 = xor(pos2 - 1, 2 * pos1 - 1)
    mask2 = xor(L - 1, 2 * pos2 - 1)

    f(ist::Int, ifn::Int, posa::Int, posb::Int, m1::Int, m2::Int, m3::Int, p::AbstractVector) = begin
        @inbounds for i in ist:ifn
            l = (4 * i & m3) | (2 * i & m2) | (i & m1) + 1
            l1 = l + posa
            l2 = l + posb
            p[l1], p[l2] = p[l2], p[l1]
        end
    end
    total_itr = div(L, 4)
    parallel_run(total_itr, Threads.nthreads(), f, pos1, pos2, mask0, mask1, mask2, v)
end

function _apply_iswap_util!(gt, v::AbstractVector, coef::Number)
    L = length(v)
    q1, q2 = ordered_positions(gt)
    pos1, pos2 = 1 << (q1-1), 1 << (q2-1)

    mask0 = pos1 - 1
    mask1 = xor(pos2 - 1, 2 * pos1 - 1)
    mask2 = xor(L - 1, 2 * pos2 - 1)

    f(ist::Int, ifn::Int, posa::Int, posb::Int, m1::Int, m2::Int, m3::Int, coeff::Number, p::AbstractVector) = begin
        @inbounds for i in ist:ifn
            l = (4 * i & m3) | (2 * i & m2) | (i & m1) + 1
            l1 = l + posa
            l2 = l + posb
            @fastmath p[l1], p[l2] = coeff*p[l2], coeff*p[l1]
        end
    end
    total_itr = div(L, 4)
    parallel_run(total_itr, Threads.nthreads(), f, pos1, pos2, mask0, mask1, mask2, convert(eltype(v), coef), v)
end

apply_threaded!(gt::iSWAPGate, v::AbstractVector) = (length(v) >= 32) ? _apply_iswap_util!(
    gt, v, im) : apply_serial!(gt, v)
apply_threaded!(gt::AdjointQuantumGate{iSWAPGate}, v::AbstractVector) = (length(v) >= 32) ? _apply_iswap_util!(
    gt, v, -im) : apply_serial!(gt, v)
