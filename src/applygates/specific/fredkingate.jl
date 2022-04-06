


function apply_threaded!(gt::FREDKINGate, v::AbstractVector)
    (length(v) < 32) && return apply_serial!(gt, v)

    L = length(v)
    q1, q2, q3 = ordered_positions(gt)
    pos1, pos2, pos3 = 1 << (q1-1), 1 << (q2-1), 1 << (q3-1)
    # stride2, stride3 = pos1 << 1, pos2 << 1
    mask0 = pos1 - 1
    mask1 = xor(pos2 - 1, 2 * pos1 - 1)
    mask2 = xor(pos3 - 1, 2 * pos2 - 1)
    mask3 = xor(L - 1, 2 * pos3 - 1)
    # println("pos1=$pos1, pos2=$pos2, m0=$mask0, m1=$mask1, m2=$mask2")

    f_f(ist::Int, ifn::Int, posa::Int, posb::Int, posc::Int, m1::Int, m2::Int, m3::Int, m4::Int, p::AbstractVector) = begin
        @inbounds for i in ist:ifn
            l000 = (8 * i & m4) | (4 * i & m3) | (2 * i & m2) | (i & m1) + 1
            # l100 = l000 + posa
            l010 = l000 + posb
            l110 = l010 + posa

            l001 = l000 + posc
            l101 = l001 + posa
            # l011 = l001 + posb
            # l111 = l011 + posa

            @fastmath p[l110], p[l101] = p[l101], p[l110]
        end
    end

    f_m(ist::Int, ifn::Int, posa::Int, posb::Int, posc::Int, m1::Int, m2::Int, m3::Int, m4::Int, p::AbstractVector) = begin
        @inbounds for i in ist:ifn
            l000 = (8 * i & m4) | (4 * i & m3) | (2 * i & m2) | (i & m1) + 1
            # l100 = l000 + posa
            l010 = l000 + posb
            l110 = l010 + posa

            l001 = l000 + posc
            # l101 = l001 + posa
            l011 = l001 + posb
            # l111 = l011 + posa

            @fastmath p[l110], p[l011] = p[l011], p[l110]
        end
    end

    f_e(ist::Int, ifn::Int, posa::Int, posb::Int, posc::Int, m1::Int, m2::Int, m3::Int, m4::Int, p::AbstractVector) = begin
        @inbounds for i in ist:ifn
            l000 = (8 * i & m4) | (4 * i & m3) | (2 * i & m2) | (i & m1) + 1
            # l100 = l000 + posa
            # l010 = l000 + posb
            # l110 = l010 + posa

            l001 = l000 + posc
            l101 = l001 + posa
            l011 = l001 + posb
            # l111 = l011 + posa

            @fastmath p[l101], p[l011] = p[l011], p[l101]
        end
    end

    target = positions(gt)[1]
    if target == q1
        f = f_f
    elseif target == q2
        f = f_m
    elseif target == q3
        f = f_e
    else
        error("target $target does not exist in $key.")
    end
    total_itr = div(L, 8)
    parallel_run(total_itr, Threads.nthreads(), f, pos1, pos2, pos3, mask0, mask1, mask2, mask3, v)
end


function apply_threaded!(gt::CCPHASEGate, v::AbstractVector)
    (length(v) < 32) && return apply_serial!(gt, v)

    L = length(v)
    q1, q2, q3 = ordered_positions(gt)
    pos1, pos2, pos3 = 1 << (q1-1), 1 << (q2-1), 1 << (q3-1)
    # stride2, stride3 = pos1 << 1, pos2 << 1
    mask0 = pos1 - 1
    mask1 = xor(pos2 - 1, 2 * pos1 - 1)
    mask2 = xor(pos3 - 1, 2 * pos2 - 1)
    mask3 = xor(L - 1, 2 * pos3 - 1)
    # println("pos1=$pos1, pos2=$pos2, m0=$mask0, m1=$mask1, m2=$mask2")

    f(ist::Int, ifn::Int, posabc1::Int, m1::Int, m2::Int, m3::Int, m4::Int, alpha::Number, p::AbstractVector) = begin
        @inbounds for i in ist:ifn
            l111 = (8 * i & m4) | (4 * i & m3) | (2 * i & m2) | (i & m1) + posabc1

            @fastmath p[l111] *= alpha
        end
    end
    exp_phi = convert(eltype(v), exp(im * parameters(gt)[1] ))

    total_itr = div(L, 8)
    parallel_run(total_itr, Threads.nthreads(), f, pos1+pos2+pos3+1, mask0, mask1, mask2, mask3, exp_phi, v)
end
