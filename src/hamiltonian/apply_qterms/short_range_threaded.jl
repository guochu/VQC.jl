


"""
    applys when key > 3, U is the transposed op
"""
function _apply_onebody_gate_H!(key::Int, U::AbstractMatrix, v::AbstractVector, vout::AbstractVector)
    L = length(v)
    sizek = 1 << (key - 1)
    mask0 = sizek - 1
    mask1 = xor(L - 1, 2 * sizek - 1)
    f(ist::Int, ifn::Int, pos::Int, m1::Int, m2::Int, mat::AbstractMatrix, p::AbstractVector, po::AbstractVector) = begin
        @inbounds for i in ist:ifn
            l = (16 * i & m2) | (8 * i & m1) + 1
            l1 = l + pos
            vi = SMatrix{8, 2}(p[l], p[l+1], p[l+2], p[l+3], p[l+4], p[l+5], p[l+6], p[l+7],
            p[l1], p[l1+1], p[l1+2], p[l1+3], p[l1+4], p[l1+5], p[l1+6], p[l1+7])

            @fastmath begin
                vo = vi * mat

                po[l] += vo[1]
                po[l+1] += vo[2]
                po[l+2] += vo[3]
                po[l+3] += vo[4]
                po[l+4] += vo[5]
                po[l+5] += vo[6]
                po[l+6] += vo[7]
                po[l+7] += vo[8]
                po[l1] += vo[9]
                po[l1+1] += vo[10]
                po[l1+2] += vo[11]
                po[l1+3] += vo[12]
                po[l1+4] += vo[13]
                po[l1+5] += vo[14]
                po[l1+6] += vo[15]
                po[l1+7] += vo[16]
            end
        end
    end

    parallel_run(div(L, 16), Threads.nthreads(), f, sizek, mask0, mask1, U, v, vout)
end

"""
    applys when key <= 3, U is the transposed op
"""
function _apply_onebody_gate_L!(key::Int, U::AbstractMatrix, v::AbstractVector, vout::AbstractVector)
    L = length(v)
    f1(ist::Int, ifn::Int, mat::AbstractMatrix, p::AbstractVector, po::AbstractVector) = begin
        @inbounds for i in ist:ifn
            l = 16 * i + 1
            vi = SMatrix{2, 8}(p[l], p[l+1], p[l+2], p[l+3], p[l+4], p[l+5], p[l+6], p[l+7],
            p[l+8], p[l+9], p[l+10], p[l+11], p[l+12], p[l+13], p[l+14], p[l+15])

            @fastmath begin
                vo = mat * vi

                po[l] += vo[1]
                po[l+1] += vo[2]
                po[l+2] += vo[3]
                po[l+3] += vo[4]
                po[l+4] += vo[5]
                po[l+5] += vo[6]
                po[l+6] += vo[7]
                po[l+7] += vo[8]
                po[l+8] += vo[9]
                po[l+9] += vo[10]
                po[l+10] += vo[11]
                po[l+11] += vo[12]
                po[l+12] += vo[13]
                po[l+13] += vo[14]
                po[l+14] += vo[15]
                po[l+15] += vo[16]
            end          

            # @fastmath po[l:(l+15)] .+= mat * vi
        end
    end
    f2(ist::Int, ifn::Int, mat::AbstractMatrix, p::AbstractVector, po::AbstractVector) = begin
        @inbounds for i in ist:ifn
            l = 16 * i + 1
            vi = SMatrix{2, 8}(p[l], p[l+2], p[l+1], p[l+3], p[l+4], p[l+6], p[l+5], p[l+7],
            p[l+8], p[l+10], p[l+9], p[l+11], p[l+12], p[l+14], p[l+13], p[l+15])

            @fastmath begin
                vo = mat * vi

                po[l] += vo[1]
                po[l+2] += vo[2]
                po[l+1] += vo[3]
                po[l+3] += vo[4]
                po[l+4] += vo[5]
                po[l+6] += vo[6]
                po[l+5] += vo[7]
                po[l+7] += vo[8]
                po[l+8] += vo[9]
                po[l+10] += vo[10]
                po[l+9] += vo[11]
                po[l+11] += vo[12]
                po[l+12] += vo[13]
                po[l+14] += vo[14]
                po[l+13] += vo[15]
                po[l+15] += vo[16]
            end

            # @fastmath p[l], p[l+2], p[l+1], p[l+3], p[l+4], p[l+6], p[l+5], p[l+7],
            # p[l+8], p[l+10], p[l+9], p[l+11], p[l+12], p[l+14], p[l+13], p[l+15] = mat * vi
        end
    end
    f3(ist::Int, ifn::Int, mat::AbstractMatrix, p::AbstractVector, po::AbstractVector) = begin
        @inbounds for i in ist:ifn
            l = 16 * i + 1
            vi = SMatrix{2, 8}(p[l], p[l+4], p[l+1], p[l+5], p[l+2], p[l+6], p[l+3], p[l+7],
            p[l+8], p[l+12], p[l+9], p[l+13], p[l+10], p[l+14], p[l+11], p[l+15])

            @fastmath begin

                vo = mat * vi

                po[l] += vo[1]
                po[l+4] += vo[2]
                po[l+1] += vo[3]
                po[l+5] += vo[4]
                po[l+2] += vo[5]
                po[l+6] += vo[6]
                po[l+3] += vo[7]
                po[l+7] += vo[8]
                po[l+8] += vo[9]
                po[l+12] += vo[10]
                po[l+9] += vo[11]
                po[l+13] += vo[12]
                po[l+10] += vo[13]
                po[l+14] += vo[14]
                po[l+11] += vo[15]
                po[l+15] += vo[16]
            end
            # @fastmath p[l], p[l+4], p[l+1], p[l+5], p[l+2], p[l+6], p[l+3], p[l+7],
            # p[l+8], p[l+12], p[l+9], p[l+13], p[l+10], p[l+14], p[l+11], p[l+15] = mat * vi
        end
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

    parallel_run(div(L, 16), Threads.nthreads(), f, U, v, vout)
end

function _apply_gate_threaded2!(q0::Int, U::AbstractMatrix, v::AbstractVector, vout::AbstractVector)
    if q0 > 3
        return _apply_onebody_gate_H!(q0, SMatrix{2,2, eltype(v)}(transpose(U)), v, vout)
    else
        return _apply_onebody_gate_L!(q0, SMatrix{2,2, eltype(v)}(U), v, vout)
    end
end

_apply_gate_threaded2!(q0::Tuple{Int}, U::AbstractMatrix, v::AbstractVector, vout::AbstractVector) = _apply_gate_threaded2!(
    q0[1], U, v, vout)

"""
    applys when both keys > 3, U is the transposed op
"""
function _apply_twobody_gate_HH!(key::Tuple{Int, Int}, U::AbstractMatrix, v::AbstractVector, vout::AbstractVector)
    L = length(v)
    q1, q2 = key
    pos1, pos2 = 1 << (q1-1), 1 << (q2-1)
    # stride2, stride3 = pos1 << 1, pos2 << 1
    mask0 = pos1 - 1
    mask1 = xor(pos2 - 1, 2 * pos1 - 1)
    mask2 = xor(L - 1, 2 * pos2 - 1)
    # println("pos1=$pos1, pos2=$pos2, m0=$mask0, m1=$mask1, m2=$mask2")

    total_itr = div(L, 32)
    f(ist::Int, ifn::Int, posa::Int, posb::Int, m1::Int, m2::Int, m3::Int, mat::AbstractMatrix, p::AbstractVector, po::AbstractVector) = begin
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

            @fastmath begin
                vo = vi * mat

                po[l] += vo[1]
                po[l+1] += vo[2]
                po[l+2] += vo[3]
                po[l+3] += vo[4]
                po[l+4] += vo[5]
                po[l+5] += vo[6]
                po[l+6] += vo[7]
                po[l+7] += vo[8]
                po[l1] += vo[9]
                po[l1+1] += vo[10]
                po[l1+2] += vo[11]
                po[l1+3] += vo[12]
                po[l1+4] += vo[13]
                po[l1+5] += vo[14]
                po[l1+6] += vo[15]
                po[l1+7] += vo[16]
                po[l2] += vo[17]
                po[l2+1] += vo[18]
                po[l2+2] += vo[19]
                po[l2+3] += vo[20]
                po[l2+4] += vo[21]
                po[l2+5] += vo[22]
                po[l2+6] += vo[23]
                po[l2+7] += vo[24]
                po[l3] += vo[25]
                po[l3+1] += vo[26]
                po[l3+2] += vo[27]
                po[l3+3] += vo[28]
                po[l3+4] += vo[29]
                po[l3+5] += vo[30]
                po[l3+6] += vo[31]
                po[l3+7] += vo[32]
            end

            # @fastmath p[l], p[l+1], p[l+2], p[l+3], p[l+4], p[l+5], p[l+6], p[l+7],
            # p[l1], p[l1+1], p[l1+2], p[l1+3], p[l1+4], p[l1+5], p[l1+6], p[l1+7],
            # p[l2], p[l2+1], p[l2+2], p[l2+3], p[l2+4], p[l2+5], p[l2+6], p[l2+7],
            # p[l3], p[l3+1], p[l3+2], p[l3+3], p[l3+4], p[l3+5], p[l3+6], p[l3+7] = vi * mat
        end
    end

    parallel_run(total_itr, Threads.nthreads(), f, pos1, pos2, mask0, mask1, mask2, U, v, vout)
end

"""
    applys when q1 <= 3 and q2 > 4, U is the transposed op
"""
function _apply_twobody_gate_LH!(key::Tuple{Int, Int}, U::AbstractMatrix, v::AbstractVector, vout::AbstractVector)
    L = length(v)
    q1, q2 = key
    sizej = 1 << (q2-1)
    mask0 = sizej - 1
    mask1 = xor(L - 1, 2 * sizej - 1)

    f1H(ist::Int, ifn::Int, posb::Int, m1::Int, m2::Int, mat::AbstractMatrix, p::AbstractVector, po::AbstractVector) = begin
        @inbounds for i in ist:ifn
            l = (32 * i & m2) | (16 * i & m1)
            l1 = l + posb
            # println("l=$l, l1=$l1")
            vi = SMatrix{8, 4}(p[l+1], p[l+3], p[l+5], p[l+7], p[l+9], p[l+11], p[l+13], p[l+15],
            p[l+2], p[l+4], p[l+6], p[l+8], p[l+10], p[l+12], p[l+14], p[l+16],
            p[l1+1], p[l1+3], p[l1+5], p[l1+7], p[l1+9], p[l1+11], p[l1+13], p[l1+15],
            p[l1+2], p[l1+4], p[l1+6], p[l1+8], p[l1+10], p[l1+12], p[l1+14], p[l1+16])

            @fastmath begin
                vo = vi * mat

                po[l+1] += vo[1]
                po[l+3] += vo[2]
                po[l+5] += vo[3]
                po[l+7] += vo[4]
                po[l+9] += vo[5]
                po[l+11] += vo[6]
                po[l+13] += vo[7]
                po[l+15] += vo[8]
                po[l+2] += vo[9]
                po[l+4] += vo[10]
                po[l+6] += vo[11]
                po[l+8] += vo[12]
                po[l+10] += vo[13]
                po[l+12] += vo[14]
                po[l+14] += vo[15]
                po[l+16] += vo[16]
                po[l1+1] += vo[17]
                po[l1+3] += vo[18]
                po[l1+5] += vo[19]
                po[l1+7] += vo[20]
                po[l1+9] += vo[21]
                po[l1+11] += vo[22]
                po[l1+13] += vo[23]
                po[l1+15] += vo[24]
                po[l1+2] += vo[25]
                po[l1+4] += vo[26]
                po[l1+6] += vo[27]
                po[l1+8] += vo[28]
                po[l1+10] += vo[29]
                po[l1+12] += vo[30]
                po[l1+14] += vo[31]
                po[l1+16] += vo[32]
            end

            # @fastmath p[l+1], p[l+3], p[l+5], p[l+7], p[l+9], p[l+11], p[l+13], p[l+15],
            # p[l+2], p[l+4], p[l+6], p[l+8], p[l+10], p[l+12], p[l+14], p[l+16],
            # p[l1+1], p[l1+3], p[l1+5], p[l1+7], p[l1+9], p[l1+11], p[l1+13], p[l1+15],
            # p[l1+2], p[l1+4], p[l1+6], p[l1+8], p[l1+10], p[l1+12], p[l1+14], p[l1+16] = vi * mat
        end
    end

    f2H(ist::Int, ifn::Int, posb::Int, m1::Int, m2::Int, mat::AbstractMatrix, p::AbstractVector, po::AbstractVector) = begin
        @inbounds for i in ist:ifn
            l = (32 * i & m2) | (16 * i & m1)
            l1 = l + posb
            # println("l=$l, l1=$l1")
            vi = SMatrix{8, 4}(p[l+1], p[l+2], p[l+5], p[l+6], p[l+9], p[l+10], p[l+13], p[l+14],
            p[l+3], p[l+4], p[l+7], p[l+8], p[l+11], p[l+12], p[l+15], p[l+16],
            p[l1+1], p[l1+2], p[l1+5], p[l1+6], p[l1+9], p[l1+10], p[l1+13], p[l1+14],
            p[l1+3], p[l1+4], p[l1+7], p[l1+8], p[l1+11], p[l1+12], p[l1+15], p[l1+16])

            @fastmath begin
                vo = vi * mat

                po[l+1] += vo[1]
                po[l+2] += vo[2]
                po[l+5] += vo[3]
                po[l+6] += vo[4]
                po[l+9] += vo[5]
                po[l+10] += vo[6]
                po[l+13] += vo[7]
                po[l+14] += vo[8]
                po[l+3] += vo[9]
                po[l+4] += vo[10]
                po[l+7] += vo[11]
                po[l+8] += vo[12]
                po[l+11] += vo[13]
                po[l+12] += vo[14]
                po[l+15] += vo[15]
                po[l+16] += vo[16]
                po[l1+1] += vo[17]
                po[l1+2] += vo[18]
                po[l1+5] += vo[19]
                po[l1+6] += vo[20]
                po[l1+9] += vo[21]
                po[l1+10] += vo[22]
                po[l1+13] += vo[23]
                po[l1+14] += vo[24]
                po[l1+3] += vo[25]
                po[l1+4] += vo[26]
                po[l1+7] += vo[27]
                po[l1+8] += vo[28]
                po[l1+11] += vo[29]
                po[l1+12] += vo[30]
                po[l1+15] += vo[31]
                po[l1+16] += vo[32]
            end

            # @fastmath p[l+1], p[l+2], p[l+5], p[l+6], p[l+9], p[l+10], p[l+13], p[l+14],
            # p[l+3], p[l+4], p[l+7], p[l+8], p[l+11], p[l+12], p[l+15], p[l+16],
            # p[l1+1], p[l1+2], p[l1+5], p[l1+6], p[l1+9], p[l1+10], p[l1+13], p[l1+14],
            # p[l1+3], p[l1+4], p[l1+7], p[l1+8], p[l1+11], p[l1+12], p[l1+15], p[l1+16] = vi * mat
        end
    end

    f3H(ist::Int, ifn::Int, posb::Int, m1::Int, m2::Int, mat::AbstractMatrix, p::AbstractVector, po::AbstractVector) = begin
        @inbounds for i in ist:ifn
            l = (32 * i & m2) | (16 * i & m1)
            l1 = l + posb
            # println("l=$l, l1=$l1")
            vi = SMatrix{8, 4}(p[l+1], p[l+2], p[l+3], p[l+4], p[l+9], p[l+10], p[l+11], p[l+12],
            p[l+5], p[l+6], p[l+7], p[l+8], p[l+13], p[l+14], p[l+15], p[l+16],
            p[l1+1], p[l1+2], p[l1+3], p[l1+4], p[l1+9], p[l1+10], p[l1+11], p[l1+12],
            p[l1+5], p[l1+6], p[l1+7], p[l1+8], p[l1+13], p[l1+14], p[l1+15], p[l1+16])

            @fastmath begin
                vo = vi * mat

                po[l+1] += vo[1]
                po[l+2] += vo[2]
                po[l+3] += vo[3]
                po[l+4] += vo[4]
                po[l+9] += vo[5]
                po[l+10] += vo[6]
                po[l+11] += vo[7]
                po[l+12] += vo[8]
                po[l+5] += vo[9]
                po[l+6] += vo[10]
                po[l+7] += vo[11]
                po[l+8] += vo[12]
                po[l+13] += vo[13]
                po[l+14] += vo[14]
                po[l+15] += vo[15]
                po[l+16] += vo[16]
                po[l1+1] += vo[17]
                po[l1+2] += vo[18]
                po[l1+3] += vo[19]
                po[l1+4] += vo[20]
                po[l1+9] += vo[21]
                po[l1+10] += vo[22]
                po[l1+11] += vo[23]
                po[l1+12] += vo[24]
                po[l1+5] += vo[25]
                po[l1+6] += vo[26]
                po[l1+7] += vo[27]
                po[l1+8] += vo[28]
                po[l1+13] += vo[29]
                po[l1+14] += vo[30]
                po[l1+15] += vo[31]
                po[l1+16] += vo[32]
            end

            # @fastmath p[l+1], p[l+2], p[l+3], p[l+4], p[l+9], p[l+10], p[l+11], p[l+12],
            # p[l+5], p[l+6], p[l+7], p[l+8], p[l+13], p[l+14], p[l+15], p[l+16],
            # p[l1+1], p[l1+2], p[l1+3], p[l1+4], p[l1+9], p[l1+10], p[l1+11], p[l1+12],
            # p[l1+5], p[l1+6], p[l1+7], p[l1+8], p[l1+13], p[l1+14], p[l1+15], p[l1+16] = vi * mat
        end
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

    parallel_run(total_itr, Threads.nthreads(), f, sizej, mask0, mask1, U, v, vout)
end


"""
    applys when both keys <= 4
"""
function _apply_twobody_gate_LL!(key::Tuple{Int, Int}, U::AbstractMatrix, v::AbstractVector, vout::AbstractVector)
    L = length(v)
    q1, q2 = key
    f12(ist::Int, ifn::Int, mat::AbstractMatrix, p::AbstractVector, po::AbstractVector) = begin
        @inbounds for i in ist:ifn
            l = 32 * i + 1
            vi = SMatrix{4, 8}(p[l], p[l+1], p[l+2], p[l+3], p[l+4], p[l+5], p[l+6], p[l+7],
            p[l+8], p[l+9], p[l+10], p[l+11], p[l+12], p[l+13], p[l+14], p[l+15],
            p[l+16], p[l+17], p[l+18], p[l+19], p[l+20], p[l+21], p[l+22], p[l+23],
            p[l+24], p[l+25], p[l+26], p[l+27], p[l+28], p[l+29], p[l+30], p[l+31])

            @fastmath begin
                vo = mat * vi

                po[l] += vo[1]
                po[l+1] += vo[2]
                po[l+2] += vo[3]
                po[l+3] += vo[4]
                po[l+4] += vo[5]
                po[l+5] += vo[6]
                po[l+6] += vo[7]
                po[l+7] += vo[8]
                po[l+8] += vo[9]
                po[l+9] += vo[10]
                po[l+10] += vo[11]
                po[l+11] += vo[12]
                po[l+12] += vo[13]
                po[l+13] += vo[14]
                po[l+14] += vo[15]
                po[l+15] += vo[16]
                po[l+16] += vo[17]
                po[l+17] += vo[18]
                po[l+18] += vo[19]
                po[l+19] += vo[20]
                po[l+20] += vo[21]
                po[l+21] += vo[22]
                po[l+22] += vo[23]
                po[l+23] += vo[24]
                po[l+24] += vo[25]
                po[l+25] += vo[26]
                po[l+26] += vo[27]
                po[l+27] += vo[28]
                po[l+28] += vo[29]
                po[l+29] += vo[30]
                po[l+30] += vo[31]
                po[l+31] += vo[32]
            end

            # @fastmath po[l:(l+31)] = mat * vi
        end
    end
    f13(ist::Int, ifn::Int, mat::AbstractMatrix, p::AbstractVector, po::AbstractVector) = begin
        @inbounds for i in ist:ifn
            l = 32 * i + 1
            vi = SMatrix{4, 8}(p[l], p[l+1], p[l+4], p[l+5], p[l+2], p[l+3], p[l+6], p[l+7],
            p[l+8], p[l+9], p[l+12], p[l+13], p[l+10], p[l+11], p[l+14], p[l+15],
            p[l+16], p[l+17], p[l+20], p[l+21], p[l+18], p[l+19], p[l+22], p[l+23],
            p[l+24], p[l+25], p[l+28], p[l+29], p[l+26], p[l+27], p[l+30], p[l+31])

            @fastmath begin
                vo = mat * vi

                po[l] += vo[1]
                po[l+1] += vo[2]
                po[l+4] += vo[3]
                po[l+5] += vo[4]
                po[l+2] += vo[5]
                po[l+3] += vo[6]
                po[l+6] += vo[7]
                po[l+7] += vo[8]
                po[l+8] += vo[9]
                po[l+9] += vo[10]
                po[l+12] += vo[11]
                po[l+13] += vo[12]
                po[l+10] += vo[13]
                po[l+11] += vo[14]
                po[l+14] += vo[15]
                po[l+15] += vo[16]
                po[l+16] += vo[17]
                po[l+17] += vo[18]
                po[l+20] += vo[19]
                po[l+21] += vo[20]
                po[l+18] += vo[21]
                po[l+19] += vo[22]
                po[l+22] += vo[23]
                po[l+23] += vo[24]
                po[l+24] += vo[25]
                po[l+25] += vo[26]
                po[l+28] += vo[27]
                po[l+29] += vo[28]
                po[l+26] += vo[29]
                po[l+27] += vo[30]
                po[l+30] += vo[31]
                po[l+31] += vo[32]
            end

            # @fastmath p[l], p[l+1], p[l+4], p[l+5], p[l+2], p[l+3], p[l+6], p[l+7],
            # p[l+8], p[l+9], p[l+12], p[l+13], p[l+10], p[l+11], p[l+14], p[l+15],
            # p[l+16], p[l+17], p[l+20], p[l+21], p[l+18], p[l+19], p[l+22], p[l+23],
            # p[l+24], p[l+25], p[l+28], p[l+29], p[l+26], p[l+27], p[l+30], p[l+31] = mat * vi
        end
    end
    f14(ist::Int, ifn::Int, mat::AbstractMatrix, p::AbstractVector, po::AbstractVector) = begin
        @inbounds for i in ist:ifn
            l = 32 * i
            vi = SMatrix{4, 8}(p[l+1], p[l+2], p[l+9], p[l+10], p[l+3], p[l+4], p[l+11], p[l+12],
            p[l+5], p[l+6], p[l+13], p[l+14], p[l+7], p[l+8], p[l+15], p[l+16],
            p[l+17], p[l+18], p[l+25], p[l+26], p[l+19], p[l+20], p[l+27], p[l+28],
            p[l+21], p[l+22], p[l+29], p[l+30], p[l+23], p[l+24], p[l+31], p[l+32])

            @fastmath begin
                vo = mat * vi

                po[l+1] += vo[1]
                po[l+2] += vo[2]
                po[l+9] += vo[3]
                po[l+10] += vo[4]
                po[l+3] += vo[5]
                po[l+4] += vo[6]
                po[l+11] += vo[7]
                po[l+12] += vo[8]
                po[l+5] += vo[9]
                po[l+6] += vo[10]
                po[l+13] += vo[11]
                po[l+14] += vo[12]
                po[l+7] += vo[13]
                po[l+8] += vo[14]
                po[l+15] += vo[15]
                po[l+16] += vo[16]
                po[l+17] += vo[17]
                po[l+18] += vo[18]
                po[l+25] += vo[19]
                po[l+26] += vo[20]
                po[l+19] += vo[21]
                po[l+20] += vo[22]
                po[l+27] += vo[23]
                po[l+28] += vo[24]
                po[l+21] += vo[25]
                po[l+22] += vo[26]
                po[l+29] += vo[27]
                po[l+30] += vo[28]
                po[l+23] += vo[29]
                po[l+24] += vo[30]
                po[l+31] += vo[31]
                po[l+32] += vo[32]
            end

            # @fastmath p[l+1], p[l+2], p[l+9], p[l+10], p[l+3], p[l+4], p[l+11], p[l+12],
            # p[l+5], p[l+6], p[l+13], p[l+14], p[l+7], p[l+8], p[l+15], p[l+16],
            # p[l+17], p[l+18], p[l+25], p[l+26], p[l+19], p[l+20], p[l+27], p[l+28],
            # p[l+21], p[l+22], p[l+29], p[l+30], p[l+23], p[l+24], p[l+31], p[l+32] = mat * vi
        end
    end
    f23(ist::Int, ifn::Int, mat::AbstractMatrix, p::AbstractVector, po::AbstractVector) = begin
        @inbounds for i in ist:ifn
            l = 32 * i + 1
            vi = SMatrix{4, 8}(p[l], p[l+2], p[l+4], p[l+6], p[l+1], p[l+3], p[l+5], p[l+7],
            p[l+8], p[l+10], p[l+12], p[l+14], p[l+9], p[l+11], p[l+13], p[l+15],
            p[l+16], p[l+18], p[l+20], p[l+22], p[l+17], p[l+19], p[l+21], p[l+23],
            p[l+24], p[l+26], p[l+28], p[l+30], p[l+25], p[l+27], p[l+29], p[l+31])

            @fastmath begin
                vo = mat * vi

                po[l] += vo[1]
                po[l+2] += vo[2]
                po[l+4] += vo[3]
                po[l+6] += vo[4]
                po[l+1] += vo[5]
                po[l+3] += vo[6]
                po[l+5] += vo[7]
                po[l+7] += vo[8]
                po[l+8] += vo[9]
                po[l+10] += vo[10]
                po[l+12] += vo[11]
                po[l+14] += vo[12]
                po[l+9] += vo[13]
                po[l+11] += vo[14]
                po[l+13] += vo[15]
                po[l+15] += vo[16]
                po[l+16] += vo[17]
                po[l+18] += vo[18]
                po[l+20] += vo[19]
                po[l+22] += vo[20]
                po[l+17] += vo[21]
                po[l+19] += vo[22]
                po[l+21] += vo[23]
                po[l+23] += vo[24]
                po[l+24] += vo[25]
                po[l+26] += vo[26]
                po[l+28] += vo[27]
                po[l+30] += vo[28]
                po[l+25] += vo[29]
                po[l+27] += vo[30]
                po[l+29] += vo[31]
                po[l+31] += vo[32]
            end

            # @fastmath p[l], p[l+2], p[l+4], p[l+6], p[l+1], p[l+3], p[l+5], p[l+7],
            # p[l+8], p[l+10], p[l+12], p[l+14], p[l+9], p[l+11], p[l+13], p[l+15],
            # p[l+16], p[l+18], p[l+20], p[l+22], p[l+17], p[l+19], p[l+21], p[l+23],
            # p[l+24], p[l+26], p[l+28], p[l+30], p[l+25], p[l+27], p[l+29], p[l+31] = mat * vi
        end
    end
    f24(ist::Int, ifn::Int, mat::AbstractMatrix, p::AbstractVector, po::AbstractVector) = begin
        @inbounds for i in ist:ifn
            l = 32 * i
            vi = SMatrix{4, 8}(p[l+1], p[l+3], p[l+9], p[l+11], p[l+2], p[l+4], p[l+10], p[l+12],
            p[l+5], p[l+7], p[l+13], p[l+15], p[l+6], p[l+8], p[l+14], p[l+16],
            p[l+17], p[l+19], p[l+25], p[l+27], p[l+18], p[l+20], p[l+26], p[l+28],
            p[l+21], p[l+23], p[l+29], p[l+31], p[l+22], p[l+24], p[l+30], p[l+32])

            @fastmath begin
                vo = mat * vi

                po[l+1] += vo[1]
                po[l+3] += vo[2]
                po[l+9] += vo[3]
                po[l+11] += vo[4]
                po[l+2] += vo[5]
                po[l+4] += vo[6]
                po[l+10] += vo[7]
                po[l+12] += vo[8]
                po[l+5] += vo[9]
                po[l+7] += vo[10]
                po[l+13] += vo[11]
                po[l+15] += vo[12]
                po[l+6] += vo[13]
                po[l+8] += vo[14]
                po[l+14] += vo[15]
                po[l+16] += vo[16]
                po[l+17] += vo[17]
                po[l+19] += vo[18]
                po[l+25] += vo[19]
                po[l+27] += vo[20]
                po[l+18] += vo[21]
                po[l+20] += vo[22]
                po[l+26] += vo[23]
                po[l+28] += vo[24]
                po[l+21] += vo[25]
                po[l+23] += vo[26]
                po[l+29] += vo[27]
                po[l+31] += vo[28]
                po[l+22] += vo[29]
                po[l+24] += vo[30]
                po[l+30] += vo[31]
                po[l+32] += vo[32]
            end

            # @fastmath p[l+1], p[l+3], p[l+9], p[l+11], p[l+2], p[l+4], p[l+10], p[l+12],
            # p[l+5], p[l+7], p[l+13], p[l+15], p[l+6], p[l+8], p[l+14], p[l+16],
            # p[l+17], p[l+19], p[l+25], p[l+27], p[l+18], p[l+20], p[l+26], p[l+28],
            # p[l+21], p[l+23], p[l+29], p[l+31], p[l+22], p[l+24], p[l+30], p[l+32] = mat * vi
        end
    end
    f34(ist::Int, ifn::Int, mat::AbstractMatrix, p::AbstractVector, po::AbstractVector) = begin
        @inbounds for i in ist:ifn
            l = 32 * i
            vi = SMatrix{4, 8}(p[l+1], p[l+5], p[l+9], p[l+13], p[l+2], p[l+6], p[l+10], p[l+14],
            p[l+3], p[l+7], p[l+11], p[l+15], p[l+4], p[l+8], p[l+12], p[l+16],
            p[l+17], p[l+21], p[l+25], p[l+29], p[l+18], p[l+22], p[l+26], p[l+30],
            p[l+19], p[l+23], p[l+27], p[l+31], p[l+20], p[l+24], p[l+28], p[l+32])

            @fastmath begin
                vo = mat * vi

                po[l+1] += vo[1]
                po[l+5] += vo[2]
                po[l+9] += vo[3]
                po[l+13] += vo[4]
                po[l+2] += vo[5]
                po[l+6] += vo[6]
                po[l+10] += vo[7]
                po[l+14] += vo[8]
                po[l+3] += vo[9]
                po[l+7] += vo[10]
                po[l+11] += vo[11]
                po[l+15] += vo[12]
                po[l+4] += vo[13]
                po[l+8] += vo[14]
                po[l+12] += vo[15]
                po[l+16] += vo[16]
                po[l+17] += vo[17]
                po[l+21] += vo[18]
                po[l+25] += vo[19]
                po[l+29] += vo[20]
                po[l+18] += vo[21]
                po[l+22] += vo[22]
                po[l+26] += vo[23]
                po[l+30] += vo[24]
                po[l+19] += vo[25]
                po[l+23] += vo[26]
                po[l+27] += vo[27]
                po[l+31] += vo[28]
                po[l+20] += vo[29]
                po[l+24] += vo[30]
                po[l+28] += vo[31]
                po[l+32] += vo[32]
            end

            # @fastmath p[l+1], p[l+5], p[l+9], p[l+13], p[l+2], p[l+6], p[l+10], p[l+14],
            # p[l+3], p[l+7], p[l+11], p[l+15], p[l+4], p[l+8], p[l+12], p[l+16],
            # p[l+17], p[l+21], p[l+25], p[l+29], p[l+18], p[l+22], p[l+26], p[l+30],
            # p[l+19], p[l+23], p[l+27], p[l+31], p[l+20], p[l+24], p[l+28], p[l+32] = mat * vi
        end
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

    parallel_run(total_itr, Threads.nthreads(), f, U, v, vout)
end

function _apply_gate_threaded2!(key::Tuple{Int, Int}, U::AbstractMatrix, v::AbstractVector, vout::AbstractVector)
    q0, q1 = key
    if q0 > 3
        return _apply_twobody_gate_HH!(key, SMatrix{4,4, eltype(v)}(transpose(U)), v, vout)
    elseif q1 > 4
        return _apply_twobody_gate_LH!(key, SMatrix{4,4, eltype(v)}(transpose(U)), v, vout)
    else
        return _apply_twobody_gate_LL!(key, SMatrix{4,4, eltype(v)}(U), v, vout)
    end
end


"""
    applys when both keys > 2, U is assumed to be transposed
"""
function _apply_threebody_gate_HHH!(key::Tuple{Int, Int, Int}, U::AbstractMatrix, v::AbstractVector, vout::AbstractVector)
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
    f(ist::Int, ifn::Int, posa::Int, posb::Int, posc::Int, m1::Int, m2::Int, m3::Int, m4::Int, mat::AbstractMatrix, p::AbstractVector, po::AbstractVector) = begin
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

            @fastmath begin
                vo = vi * mat

                po[l000] += vo[1]
                po[l000+1] += vo[2]
                po[l000+2] += vo[3]
                po[l000+3] += vo[4]
                po[l100] += vo[5]
                po[l100+1] += vo[6]
                po[l100+2] += vo[7]
                po[l100+3] += vo[8]
                po[l010] += vo[9]
                po[l010+1] += vo[10]
                po[l010+2] += vo[11]
                po[l010+3] += vo[12]
                po[l110] += vo[13] 
                po[l110+1] += vo[14] 
                po[l110+2] += vo[15] 
                po[l110+3] += vo[16]
                po[l001] += vo[17] 
                po[l001+1] += vo[18] 
                po[l001+2] += vo[19] 
                po[l001+3] += vo[20]
                po[l101] += vo[21] 
                po[l101+1] += vo[22] 
                po[l101+2] += vo[23] 
                po[l101+3] += vo[24]
                po[l011] += vo[25] 
                po[l011+1] += vo[26] 
                po[l011+2] += vo[27] 
                po[l011+3] += vo[28]
                po[l111] += vo[29] 
                po[l111+1] += vo[30] 
                po[l111+2] += vo[31]  
                po[l111+3] += vo[32]
            end

            # @fastmath p[l000], p[l000+1], p[l000+2], p[l000+3],
            #           p[l100], p[l100+1], p[l100+2], p[l100+3],
            #           p[l010], p[l010+1], p[l010+2], p[l010+3],
            #           p[l110], p[l110+1], p[l110+2], p[l110+3],
            #           p[l001], p[l001+1], p[l001+2], p[l001+3],
            #           p[l101], p[l101+1], p[l101+2], p[l101+3],
            #           p[l011], p[l011+1], p[l011+2], p[l011+3],
            #           p[l111], p[l111+1], p[l111+2], p[l111+3] = vi * mat
        end
    end

    parallel_run(total_itr, Threads.nthreads(), f, pos1, pos2, pos3, mask0, mask1, mask2, mask3, U, v, vout)
end

"""
    applys when q1 <= 2 and q2 > 3, U is the transposed op
"""
function _apply_threebody_gate_LHH!(key::Tuple{Int, Int, Int}, U::AbstractMatrix, v::AbstractVector, vout::AbstractVector)
    L = length(v)
    q1, q2, q3 = key
    sizej, sizel = 1 << (q2-1), 1 << (q3-1)
    mask0 = sizej - 1
    mask1 = xor(sizel - 1, 2 * sizej - 1)
    mask2 = xor(L-1, 2 * sizel - 1)

    f1H(ist::Int, ifn::Int, posb::Int, posc::Int, m1::Int, m2::Int, m3::Int, mat::AbstractMatrix, p::AbstractVector, po::AbstractVector) = begin
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

            @fastmath begin
                vo = vi * mat

                po[l+1] += vo[1]
                po[l+3] += vo[2] 
                po[l+5] += vo[3] 
                po[l+7] += vo[4]
                po[l+2] += vo[5] 
                po[l+4] += vo[6] 
                po[l+6] += vo[7] 
                po[l+8] += vo[8]
                po[l1+1] += vo[9] 
                po[l1+3] += vo[10] 
                po[l1+5] += vo[11] 
                po[l1+7] += vo[12]
                po[l1+2] += vo[13] 
                po[l1+4] += vo[14] 
                po[l1+6] += vo[15] 
                po[l1+8] += vo[16]
                po[l2+1] += vo[17] 
                po[l2+3] += vo[18] 
                po[l2+5] += vo[19] 
                po[l2+7] += vo[20]
                po[l2+2] += vo[21] 
                po[l2+4] += vo[22] 
                po[l2+6] += vo[23] 
                po[l2+8] += vo[24]
                po[l3+1] += vo[25] 
                po[l3+3] += vo[26] 
                po[l3+5] += vo[27] 
                po[l3+7] += vo[28]
                po[l3+2] += vo[29] 
                po[l3+4] += vo[30] 
                po[l3+6] += vo[31] 
                po[l3+8] += vo[32]
            end

            # @fastmath p[l+1], p[l+3], p[l+5], p[l+7],
            #           p[l+2], p[l+4], p[l+6], p[l+8],
            #           p[l1+1], p[l1+3], p[l1+5], p[l1+7],
            #           p[l1+2], p[l1+4], p[l1+6], p[l1+8],
            #           p[l2+1], p[l2+3], p[l2+5], p[l2+7],
            #           p[l2+2], p[l2+4], p[l2+6], p[l2+8],
            #           p[l3+1], p[l3+3], p[l3+5], p[l3+7],
            #           p[l3+2], p[l3+4], p[l3+6], p[l3+8] = vi * mat
        end
    end

    f2H(ist::Int, ifn::Int, posb::Int, posc::Int, m1::Int, m2::Int, m3::Int, mat::AbstractMatrix, p::AbstractVector, po::AbstractVector) = begin
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

            @fastmath begin
                vo = vi * mat

                po[l+1] += vo[1] 
                po[l+2] += vo[2] 
                po[l+5] += vo[3] 
                po[l+6] += vo[4]
                po[l+3] += vo[5] 
                po[l+4] += vo[6] 
                po[l+7] += vo[7] 
                po[l+8] += vo[8]
                po[l1+1] += vo[9] 
                po[l1+2] += vo[10] 
                po[l1+5] += vo[11] 
                po[l1+6] += vo[12]
                po[l1+3] += vo[13] 
                po[l1+4] += vo[14] 
                po[l1+7] += vo[15] 
                po[l1+8] += vo[16]
                po[l2+1] += vo[17] 
                po[l2+2] += vo[18] 
                po[l2+5] += vo[19] 
                po[l2+6] += vo[20]
                po[l2+3] += vo[21] 
                po[l2+4] += vo[22] 
                po[l2+7] += vo[23] 
                po[l2+8] += vo[24]
                po[l3+1] += vo[25] 
                po[l3+2] += vo[26] 
                po[l3+5] += vo[27] 
                po[l3+6] += vo[28]
                po[l3+3] += vo[29] 
                po[l3+4] += vo[30] 
                po[l3+7] += vo[31] 
                po[l3+8] += vo[32]
            end

            # @fastmath p[l+1], p[l+2], p[l+5], p[l+6],
            #           p[l+3], p[l+4], p[l+7], p[l+8],
            #           p[l1+1], p[l1+2], p[l1+5], p[l1+6],
            #           p[l1+3], p[l1+4], p[l1+7], p[l1+8],
            #           p[l2+1], p[l2+2], p[l2+5], p[l2+6],
            #           p[l2+3], p[l2+4], p[l2+7], p[l2+8],
            #           p[l3+1], p[l3+2], p[l3+5], p[l3+6],
            #           p[l3+3], p[l3+4], p[l3+7], p[l3+8] = vi * mat
        end
    end

    f3H(ist::Int, ifn::Int, posb::Int, posc::Int, m1::Int, m2::Int, m3::Int, mat::AbstractMatrix, p::AbstractVector, po::AbstractVector) = begin
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

            @fastmath begin
                vo = vi * mat

                po[l+1] += vo[1] 
                po[l+2] += vo[2] 
                po[l+3] += vo[3] 
                po[l+4] += vo[4]
                po[l+5] += vo[5] 
                po[l+6] += vo[6] 
                po[l+7] += vo[7] 
                po[l+8] += vo[8]
                po[l1+1] += vo[9] 
                po[l1+2] += vo[10] 
                po[l1+3] += vo[11] 
                po[l1+4] += vo[12]
                po[l1+5] += vo[13] 
                po[l1+6] += vo[14] 
                po[l1+7] += vo[15] 
                po[l1+8] += vo[16]
                po[l2+1] += vo[17] 
                po[l2+2] += vo[18] 
                po[l2+3] += vo[19] 
                po[l2+4] += vo[20]
                po[l2+5] += vo[21] 
                po[l2+6] += vo[22] 
                po[l2+7] += vo[23] 
                po[l2+8] += vo[24]
                po[l3+1] += vo[25] 
                po[l3+2] += vo[26] 
                po[l3+3] += vo[27] 
                po[l3+4] += vo[28]
                po[l3+5] += vo[29] 
                po[l3+6] += vo[30] 
                po[l3+7] += vo[31] 
                po[l3+8] += vo[32]
            end

            # @fastmath p[l+1], p[l+2], p[l+3], p[l+4],
            #           p[l+5], p[l+6], p[l+7], p[l+8],
            #           p[l1+1], p[l1+2], p[l1+3], p[l1+4],
            #           p[l1+5], p[l1+6], p[l1+7], p[l1+8],
            #           p[l2+1], p[l2+2], p[l2+3], p[l2+4],
            #           p[l2+5], p[l2+6], p[l2+7], p[l2+8],
            #           p[l3+1], p[l3+2], p[l3+3], p[l3+4],
            #           p[l3+5], p[l3+6], p[l3+7], p[l3+8] = vi * mat
        end
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

    parallel_run(total_itr, Threads.nthreads(), f, sizej, sizel, mask0, mask1, mask2, U, v, vout)
end

"""
    applys when q1, q2 <= 3 and q3 > 4, U is the transposed op
"""
function _apply_threebody_gate_LLH!(key::Tuple{Int, Int, Int}, U::AbstractMatrix, v::AbstractVector, vout::AbstractVector)
    L = length(v)
    q1, q2, q3 = key
    sizej = 1 << (q3-1)
    mask0 = sizej - 1
    mask1 = xor(L - 1, 2 * sizej - 1)

    f12H(ist::Int, ifn::Int, posb::Int, m1::Int, m2::Int, mat::AbstractMatrix, p::AbstractVector, po::AbstractVector) = begin
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

            @fastmath begin
                vo = vi * mat

                po[l+1] += vo[1] 
                po[l+5] += vo[2] 
                po[l+9] += vo[3] 
                po[l+13] += vo[4]
                po[l+2] += vo[5] 
                po[l+6] += vo[6] 
                po[l+10] += vo[7] 
                po[l+14] += vo[8]
                po[l+3] += vo[9] 
                po[l+7] += vo[10] 
                po[l+11] += vo[11] 
                po[l+15] += vo[12]
                po[l+4] += vo[13] 
                po[l+8] += vo[14] 
                po[l+12] += vo[15] 
                po[l+16] += vo[16]
                po[l1+1] += vo[17]
                po[l1+5] += vo[18] 
                po[l1+9] += vo[19] 
                po[l1+13] += vo[20]
                po[l1+2] += vo[21] 
                po[l1+6] += vo[22] 
                po[l1+10] += vo[23] 
                po[l1+14] += vo[24]
                po[l1+3] += vo[25] 
                po[l1+7] += vo[26] 
                po[l1+11] += vo[27] 
                po[l1+15] += vo[28]
                po[l1+4] += vo[29] 
                po[l1+8] += vo[30] 
                po[l1+12] += vo[31] 
                po[l1+16] += vo[32]
            end

            # @fastmath p[l+1], p[l+5], p[l+9], p[l+13],
            #           p[l+2], p[l+6], p[l+10], p[l+14],
            #           p[l+3], p[l+7], p[l+11], p[l+15],
            #           p[l+4], p[l+8], p[l+12], p[l+16],
            #           p[l1+1], p[l1+5], p[l1+9], p[l1+13],
            #           p[l1+2], p[l1+6], p[l1+10], p[l1+14],
            #           p[l1+3], p[l1+7], p[l1+11], p[l1+15],
            #           p[l1+4], p[l1+8], p[l1+12], p[l1+16] = vi * mat
        end
    end

    f13H(ist::Int, ifn::Int, posb::Int, m1::Int, m2::Int, mat::AbstractMatrix, p::AbstractVector, po::AbstractVector) = begin
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

            @fastmath begin
                vo = vi * mat

                po[l+1] += vo[1] 
                po[l+3] += vo[2] 
                po[l+9] += vo[3] 
                po[l+11] += vo[4]
                po[l+2] += vo[5] 
                po[l+4] += vo[6] 
                po[l+10] += vo[7] 
                po[l+12] += vo[8]
                po[l+5] += vo[9] 
                po[l+7] += vo[10] 
                po[l+13] += vo[11] 
                po[l+15] += vo[12]
                po[l+6] += vo[13] 
                po[l+8] += vo[14] 
                po[l+14] += vo[15] 
                po[l+16] += vo[16]
                po[l1+1] += vo[17] 
                po[l1+3] += vo[18] 
                po[l1+9] += vo[19] 
                po[l1+11] += vo[20]
                po[l1+2] += vo[21] 
                po[l1+4] += vo[22] 
                po[l1+10] += vo[23] 
                po[l1+12] += vo[24]
                po[l1+5] += vo[25] 
                po[l1+7] += vo[26] 
                po[l1+13] += vo[27] 
                po[l1+15] += vo[28]
                po[l1+6] += vo[29] 
                po[l1+8] += vo[30] 
                po[l1+14] += vo[31] 
                po[l1+16] += vo[32]
            end

            # @fastmath p[l+1], p[l+3], p[l+9], p[l+11],
            #           p[l+2], p[l+4], p[l+10], p[l+12],
            #           p[l+5], p[l+7], p[l+13], p[l+15],
            #           p[l+6], p[l+8], p[l+14], p[l+16],
            #           p[l1+1], p[l1+3], p[l1+9], p[l1+11],
            #           p[l1+2], p[l1+4], p[l1+10], p[l1+12],
            #           p[l1+5], p[l1+7], p[l1+13], p[l1+15],
            #           p[l1+6], p[l1+8], p[l1+14], p[l1+16] = vi * mat
        end
    end

    f23H(ist::Int, ifn::Int, posb::Int, m1::Int, m2::Int, mat::AbstractMatrix, p::AbstractVector, po::AbstractVector) = begin
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

            @fastmath begin
                vo = vi * mat

                po[l+1] += vo[1] 
                po[l+2] += vo[2] 
                po[l+9] += vo[3] 
                po[l+10] += vo[4]
                po[l+3] += vo[5] 
                po[l+4] += vo[6] 
                po[l+11] += vo[7] 
                po[l+12] += vo[8]
                po[l+5] += vo[9] 
                po[l+6] += vo[10] 
                po[l+13] += vo[11] 
                po[l+14] += vo[12]
                po[l+7] += vo[13] 
                po[l+8] += vo[14] 
                po[l+15] += vo[15] 
                po[l+16] += vo[16]
                po[l1+1] += vo[17] 
                po[l1+2] += vo[18] 
                po[l1+9] += vo[19] 
                po[l1+10] += vo[20]
                po[l1+3] += vo[21] 
                po[l1+4] += vo[22] 
                po[l1+11] += vo[23] 
                po[l1+12] += vo[24]
                po[l1+5] += vo[25] 
                po[l1+6] += vo[26] 
                po[l1+13] += vo[27] 
                po[l1+14] += vo[28]
                po[l1+7] += vo[29] 
                po[l1+8] += vo[30] 
                po[l1+15] += vo[31]
                po[l1+16] += vo[32]     
            end

            # @fastmath p[l+1], p[l+2], p[l+9], p[l+10],
            #           p[l+3], p[l+4], p[l+11], p[l+12],
            #           p[l+5], p[l+6], p[l+13], p[l+14],
            #           p[l+7], p[l+8], p[l+15], p[l+16],
            #           p[l1+1], p[l1+2], p[l1+9], p[l1+10],
            #           p[l1+3], p[l1+4], p[l1+11], p[l1+12],
            #           p[l1+5], p[l1+6], p[l1+13], p[l1+14],
            #           p[l1+7], p[l1+8], p[l1+15], p[l1+16] = vi * mat
        end
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

    parallel_run(total_itr, Threads.nthreads(), f, sizej, mask0, mask1, U, v, vout)
end

"""
    applys when both keys <= 4
"""
function _apply_threebody_gate_LLL!(key::Tuple{Int, Int, Int}, U::AbstractMatrix, v::AbstractVector, vout::AbstractVector)
    L = length(v)
    q1, q2, q3 = key
    f123(ist::Int, ifn::Int, mat::AbstractMatrix, p::AbstractVector, po::AbstractVector) = begin
        @inbounds for i in ist:ifn
            l = 32 * i + 1
            vi = SMatrix{8, 4}(p[l], p[l+1], p[l+2], p[l+3], p[l+4], p[l+5], p[l+6], p[l+7],
            p[l+8], p[l+9], p[l+10], p[l+11], p[l+12], p[l+13], p[l+14], p[l+15],
            p[l+16], p[l+17], p[l+18], p[l+19], p[l+20], p[l+21], p[l+22], p[l+23],
            p[l+24], p[l+25], p[l+26], p[l+27], p[l+28], p[l+29], p[l+30], p[l+31])

            @fastmath begin
                vo = mat * vi

                po[l] += vo[1]
                po[l+1] += vo[2]
                po[l+2] += vo[3]
                po[l+3] += vo[4]
                po[l+4] += vo[5]
                po[l+5] += vo[6]
                po[l+6] += vo[7]
                po[l+7] += vo[8]
                po[l+8] += vo[9]
                po[l+9] += vo[10]
                po[l+10] += vo[11]
                po[l+11] += vo[12]
                po[l+12] += vo[13]
                po[l+13] += vo[14]
                po[l+14] += vo[15]
                po[l+15] += vo[16]
                po[l+16] += vo[17]
                po[l+17] += vo[18]
                po[l+18] += vo[19]
                po[l+19] += vo[20]
                po[l+20] += vo[21]
                po[l+21] += vo[22]
                po[l+22] += vo[23]
                po[l+23] += vo[24]
                po[l+24] += vo[25]
                po[l+25] += vo[26]
                po[l+26] += vo[27]
                po[l+27] += vo[28]
                po[l+28] += vo[29]
                po[l+29] += vo[30]
                po[l+30] += vo[31]
                po[l+31] += vo[32]               
            end
            # @fastmath p[l:(l+31)]= mat * vi
        end
    end
    f124(ist::Int, ifn::Int, mat::AbstractMatrix, p::AbstractVector, po::AbstractVector) = begin
        @inbounds for i in ist:ifn
            l = 32 * i
            vi = SMatrix{8, 4}(p[l+1], p[l+2], p[l+3], p[l+4], p[l+9], p[l+10], p[l+11], p[l+12],
                               p[l+5], p[l+6], p[l+7], p[l+8], p[l+13], p[l+14], p[l+15], p[l+16],
                               p[l+17], p[l+18], p[l+19], p[l+20], p[l+25], p[l+26], p[l+27], p[l+28],
                               p[l+21], p[l+22], p[l+23], p[l+24], p[l+29], p[l+30], p[l+31], p[l+32])

            @fastmath begin
                vo = mat * vi

                po[l+1] += vo[1] 
                po[l+2] += vo[2] 
                po[l+3] += vo[3] 
                po[l+4] += vo[4] 
                po[l+9] += vo[5] 
                po[l+10] += vo[6] 
                po[l+11] += vo[7] 
                po[l+12] += vo[8]
                po[l+5] += vo[9] 
                po[l+6] += vo[10] 
                po[l+7] += vo[11] 
                po[l+8] += vo[12] 
                po[l+13] += vo[13] 
                po[l+14] += vo[14] 
                po[l+15] += vo[15] 
                po[l+16] += vo[16]
                po[l+17] += vo[17] 
                po[l+18] += vo[18] 
                po[l+19] += vo[19] 
                po[l+20] += vo[20] 
                po[l+25] += vo[21] 
                po[l+26] += vo[22] 
                po[l+27] += vo[23] 
                po[l+28] += vo[24]
                po[l+21] += vo[25] 
                po[l+22] += vo[26] 
                po[l+23] += vo[27] 
                po[l+24] += vo[28] 
                po[l+29] += vo[29] 
                po[l+30] += vo[30] 
                po[l+31] += vo[31] 
                po[l+32] += vo[32]    
            end

            # @fastmath p[l+1], p[l+2], p[l+3], p[l+4], p[l+9], p[l+10], p[l+11], p[l+12],
            #           p[l+5], p[l+6], p[l+7], p[l+8], p[l+13], p[l+14], p[l+15], p[l+16],
            #           p[l+17], p[l+18], p[l+19], p[l+20], p[l+25], p[l+26], p[l+27], p[l+28],
            #           p[l+21], p[l+22], p[l+23], p[l+24], p[l+29], p[l+30], p[l+31], p[l+32] = mat * vi
        end
    end
    f134(ist::Int, ifn::Int, mat::AbstractMatrix, p::AbstractVector, po::AbstractVector) = begin
        @inbounds for i in ist:ifn
            l = 32 * i
            vi = SMatrix{8, 4}(p[l+1], p[l+2], p[l+5], p[l+6], p[l+9], p[l+10], p[l+13], p[l+14],
                               p[l+3], p[l+4], p[l+7], p[l+8], p[l+11], p[l+12], p[l+15], p[l+16],
                               p[l+17], p[l+18], p[l+21], p[l+22], p[l+25], p[l+26], p[l+29], p[l+30],
                               p[l+19], p[l+20], p[l+23], p[l+24], p[l+27], p[l+28], p[l+31], p[l+32])

            @fastmath begin
                vo = mat * vi

                po[l+1] += vo[1] 
                po[l+2] += vo[2] 
                po[l+5] += vo[3] 
                po[l+6] += vo[4] 
                po[l+9] += vo[5] 
                po[l+10] += vo[6] 
                po[l+13] += vo[7] 
                po[l+14] += vo[8]
                po[l+3] += vo[9] 
                po[l+4] += vo[10] 
                po[l+7] += vo[11] 
                po[l+8] += vo[12] 
                po[l+11] += vo[13] 
                po[l+12] += vo[14] 
                po[l+15] += vo[15] 
                po[l+16] += vo[16]
                po[l+17] += vo[17] 
                po[l+18] += vo[18] 
                po[l+21] += vo[19] 
                po[l+22] += vo[20] 
                po[l+25] += vo[21] 
                po[l+26] += vo[22] 
                po[l+29] += vo[23] 
                po[l+30] += vo[24]
                po[l+19] += vo[25] 
                po[l+20] += vo[26] 
                po[l+23] += vo[27] 
                po[l+24] += vo[28] 
                po[l+27] += vo[29] 
                po[l+28] += vo[30] 
                po[l+31] += vo[31] 
                po[l+32] += vo[32]
            end

            # @fastmath p[l+1], p[l+2], p[l+5], p[l+6], p[l+9], p[l+10], p[l+13], p[l+14],
            #           p[l+3], p[l+4], p[l+7], p[l+8], p[l+11], p[l+12], p[l+15], p[l+16],
            #           p[l+17], p[l+18], p[l+21], p[l+22], p[l+25], p[l+26], p[l+29], p[l+30],
            #           p[l+19], p[l+20], p[l+23], p[l+24], p[l+27], p[l+28], p[l+31], p[l+32] = mat * vi
        end
    end
    f234(ist::Int, ifn::Int, mat::AbstractMatrix, p::AbstractVector, po::AbstractVector) = begin
        @inbounds for i in ist:ifn
            l = 32 * i
            vi = SMatrix{8, 4}(p[l+1], p[l+3], p[l+5], p[l+7], p[l+9], p[l+11], p[l+13], p[l+15],
                               p[l+2], p[l+4], p[l+6], p[l+8], p[l+10], p[l+12], p[l+14], p[l+16],
                               p[l+17], p[l+19], p[l+21], p[l+23], p[l+25], p[l+27], p[l+29], p[l+31],
                               p[l+18], p[l+20], p[l+22], p[l+24], p[l+26], p[l+28], p[l+30], p[l+32])

            @fastmath begin
                vo = mat * vi
                
                po[l+1] += vo[1] 
                po[l+3] += vo[2] 
                po[l+5] += vo[3] 
                po[l+7] += vo[4] 
                po[l+9] += vo[5] 
                po[l+11] += vo[6] 
                po[l+13] += vo[7]
                po[l+15] += vo[8]
                po[l+2] += vo[9] 
                po[l+4] += vo[10] 
                po[l+6] += vo[11] 
                po[l+8] += vo[12] 
                po[l+10] += vo[13] 
                po[l+12] += vo[14] 
                po[l+14] += vo[15] 
                po[l+16] += vo[16]
                po[l+17] += vo[17] 
                po[l+19] += vo[18] 
                po[l+21] += vo[19] 
                po[l+23] += vo[20] 
                po[l+25] += vo[21] 
                po[l+27] += vo[22] 
                po[l+29] += vo[23] 
                po[l+31] += vo[24]
                po[l+18] += vo[25] 
                po[l+20] += vo[26] 
                po[l+22] += vo[27] 
                po[l+24] += vo[28] 
                po[l+26] += vo[29] 
                po[l+28] += vo[30] 
                po[l+30] += vo[31] 
                po[l+32] += vo[32]    
            end

            # @fastmath p[l+1], p[l+3], p[l+5], p[l+7], p[l+9], p[l+11], p[l+13], p[l+15],
            #           p[l+2], p[l+4], p[l+6], p[l+8], p[l+10], p[l+12], p[l+14], p[l+16],
            #           p[l+17], p[l+19], p[l+21], p[l+23], p[l+25], p[l+27], p[l+29], p[l+31],
            #           p[l+18], p[l+20], p[l+22], p[l+24], p[l+26], p[l+28], p[l+30], p[l+32] = mat * vi
        end
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

    parallel_run(total_itr, Threads.nthreads(), f, U, v, vout)
end

function _apply_gate_threaded2!(key::Tuple{Int, Int, Int}, U::AbstractMatrix, v::AbstractVector, vout::AbstractVector)
    q0, q1, q2 = key
    if q0 > 2
        return _apply_threebody_gate_HHH!(key, SMatrix{8,8, eltype(v)}(transpose(U)), v, vout)
    elseif q1 > 3
        return _apply_threebody_gate_LHH!(key, SMatrix{8,8, eltype(v)}(transpose(U)), v, vout)
    elseif q2 > 4
        return _apply_threebody_gate_LLH!(key, SMatrix{8,8, eltype(v)}(transpose(U)), v, vout)
    else
        return _apply_threebody_gate_LLL!(key, SMatrix{8,8, eltype(v)}(U), v, vout)
    end
end


# assume vout is initialized with 0s
function _apply_threaded_util!(m::QubitsOperator, v::AbstractVector, vout::AbstractVector)
    for (k, bond) in m.data
        _apply_gate_threaded2!(k, _get_mat(length(k), bond), v, vout)
    end
    return vout
end



