

"""
    specialization for PHASEGate
"""
function apply_threaded!(gt::PHASEGate, v::AbstractVector)
    (length(v) < 32) && return apply_serial!(gt, v)
    L = length(v)
    sizek = 1 << (ordered_positions(gt)[1] - 1)
    mask0 = sizek - 1
    mask1 = xor(L - 1, 2 * sizek - 1)
    f(ist::Int, ifn::Int, pos::Int, m1::Int, m2::Int, alpha::Number, p::AbstractVector) = begin
        @inbounds for i in ist:ifn
            l = (2 * i & m2) | (i & m1) + pos
            @fastmath p[l] *= alpha
        end
    end

    exp_phi = convert(eltype(v), exp(im * parameters(gt)[1] ))
    total_itr = div(L, 2)
    parallel_run(total_itr, Threads.nthreads(), f, sizek+1, mask0, mask1, exp_phi, v)
end

"""
    specialization for XGate
"""
function apply_threaded!(gt::XGate, v::AbstractVector)
    (length(v) < 32) && return apply_serial!(gt, v)
    L = length(v)
    sizek = 1 << (ordered_positions(gt)[1] - 1)
    mask0 = sizek - 1
    mask1 = xor(L - 1, 2 * sizek - 1)
    f(ist::Int, ifn::Int, pos::Int, m1::Int, m2::Int, p::AbstractVector) = begin
        @inbounds for i in ist:ifn
            l = (2 * i & m2) | (i & m1) + 1
            l1 = l + pos
            @fastmath p[l], p[l1] = p[l1], p[l]
        end
    end

    total_itr = div(L, 2)
    parallel_run(total_itr, Threads.nthreads(), f, sizek, mask0, mask1, v)
end
