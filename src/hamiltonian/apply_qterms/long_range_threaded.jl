

function _apply_fourbody_term_impl!(key::Tuple{Int, Int, Int, Int}, U::AbstractMatrix, v::AbstractVector, vout::AbstractVector)
    L = length(v)
    q1, q2, q3, q4 = key
    pos1, pos2, pos3, pos4 = 1 << (q1-1), 1 << (q2-1), 1 << (q3-1), 1 << (q4-1)
    # stride2, stride3 = pos1 << 1, pos2 << 1
    mask0 = pos1 - 1
    mask1 = xor(pos2 - 1, 2 * pos1 - 1)
    mask2 = xor(pos3 - 1, 2 * pos2 - 1)
    mask3 = xor(pos4 - 1, 2 * pos3 - 1)
    mask4 = xor(L - 1, 2 * pos4 - 1)
    # println("pos1=$pos1, pos2=$pos2, m0=$mask0, m1=$mask1, m2=$mask2")

    
    function f(ist::Int, ifn::Int, posa::Int, posb::Int, posc::Int, posd::Int, m1::Int, m2::Int, m3::Int, 
    	m4::Int, m5::Int, mat::AbstractMatrix, p::AbstractVector, po::AbstractVector) 
        @inbounds for i in ist:ifn
            l0000 = (16 * i & m5) | (8 * i & m4) | (4 * i & m3) | (2 * i & m2) | (i & m1) + 1
            l1000 = l0000 + posa
            l0100 = l0000 + posb
            l0010 = l0000 + posc
            l0001 = l0000 + posd

            l1100 = l0100 + posa
            l1010 = l0010 + posa
            l1001 = l1000 + posd
            l0110 = l0010 + posb
            l0101 = l0100 + posd
            l0011 = l0010 + posd
            
            l1110 = l1100 + posc
            l1101 = l1100 + posd
            l1011 = l1010 + posd
            l0111 = l0110 + posd

            l1111 = l1110 + posd

            vi = SVector(p[l0000], p[l1000], p[l0100], p[l1100], p[l0010], p[l1010], p[l0110], p[l1110],
                p[l0001], p[l1001], p[l0101], p[l1101], p[l0011], p[l1011], p[l0111], p[l1111])

            @fastmath begin
                vo = mat * vi
                po[l0000] += vo[1] 
                po[l1000] += vo[2] 
                po[l0100] += vo[3] 
                po[l1100] += vo[4] 
                po[l0010] += vo[5] 
                po[l1010] += vo[6] 
                po[l0110] += vo[7] 
                po[l1110] += vo[8]
                po[l0001] += vo[9] 
                po[l1001] += vo[10] 
                po[l0101] += vo[11] 
                po[l1101] += vo[12] 
                po[l0011] += vo[13] 
                po[l1011] += vo[14] 
                po[l0111] += vo[15] 
                po[l1111] += vo[16]
            end
        end
    end
    total_itr = div(L, 16)
    parallel_run(total_itr, Threads.nthreads(), f, pos1, pos2, pos3, pos4, mask0, mask1, mask2, mask3, mask4, U, v, vout)
end


_apply_gate_threaded2!(key::Tuple{Int, Int, Int, Int}, U::AbstractMatrix, v::AbstractVector, vout::AbstractVector) = _apply_fourbody_term_impl!(
	key, SMatrix{16,16, eltype(v)}(U), v, vout)


_apply_gate_2!(key::Tuple{Int, Int, Int, Int}, U::AbstractMatrix, v::AbstractVector, vout::AbstractVector) = _apply_gate_threaded2!(
	key, U, v, vout)

function _apply_fivebody_term_impl!(key::Tuple{Int, Int, Int, Int, Int}, U::AbstractMatrix, v::AbstractVector, vout::AbstractVector)
    L = length(v)
    q1, q2, q3, q4, q5 = key
    pos1, pos2, pos3, pos4, pos5 = 1 << (q1-1), 1 << (q2-1), 1 << (q3-1), 1 << (q4-1), 1 << (q5-1)
    # stride2, stride3 = pos1 << 1, pos2 << 1
    mask0 = pos1 - 1
    mask1 = xor(pos2 - 1, 2 * pos1 - 1)
    mask2 = xor(pos3 - 1, 2 * pos2 - 1)
    mask3 = xor(pos4 - 1, 2 * pos3 - 1)
    mask4 = xor(pos5 - 1, 2 * pos4 - 1)
    mask5 = xor(L - 1, 2 * pos5 - 1)
    # println("pos1=$pos1, pos2=$pos2, m0=$mask0, m1=$mask1, m2=$mask2")

    function f(ist::Int, ifn::Int, posa::Int, posb::Int, posc::Int, posd::Int, pose::Int, 
        m1::Int, m2::Int, m3::Int, m4::Int, m5::Int, m6::Int, mat::AbstractMatrix, p::AbstractVector, po::AbstractVector) 
        @inbounds for i in ist:ifn
            l00000 = (32 * i & m6) | (16 * i & m5) | (8 * i & m4) | (4 * i & m3) | (2 * i & m2) | (i & m1) + 1
            l10000 = l00000 + posa
            l01000 = l00000 + posb
            l00100 = l00000 + posc
            l00010 = l00000 + posd
            l00001 = l00000 + pose

            l11000 = l01000 + posa
            l10100 = l00100 + posa
            l10010 = l10000 + posd
            l10001 = l10000 + pose
            l01100 = l00100 + posb
            l01010 = l01000 + posd
            l01001 = l01000 + pose
            l00110 = l00100 + posd
            l00101 = l00100 + pose
            l00011 = l00010 + pose

            
            l11100 = l11000 + posc
            l11010 = l11000 + posd
            l11001 = l11000 + pose
            l10110 = l10100 + posd
            l10101 = l10100 + pose
            l10011 = l10010 + pose
            l01110 = l01100 + posd
            l01101 = l01100 + pose
            l01011 = l01010 + pose
            l00111 = l00110 + pose

            l11110 = l11100 + posd
            l11101 = l11100 + pose
            l11011 = l11010 + pose
            l10111 = l10110 + pose
            l01111 = l01110 + pose

            l11111 = l11110 + pose

            vi = SVector(p[l00000], p[l10000], p[l01000], p[l11000], p[l00100], p[l10100], p[l01100], p[l11100],
                p[l00010], p[l10010], p[l01010], p[l11010], p[l00110], p[l10110], p[l01110], p[l11110],
                p[l00001], p[l10001], p[l01001], p[l11001], p[l00101], p[l10101], p[l01101], p[l11101],
                p[l00011], p[l10011], p[l01011], p[l11011], p[l00111], p[l10111], p[l01111], p[l11111])

            @fastmath begin
                vo = mat * vi

                po[l00000] += vo[1] 
                po[l10000] += vo[2] 
                po[l01000] += vo[3] 
                po[l11000] += vo[4] 
                po[l00100] += vo[5] 
                po[l10100] += vo[6] 
                po[l01100] += vo[7] 
                po[l11100] += vo[8]
                po[l00010] += vo[9] 
                po[l10010] += vo[10] 
                po[l01010] += vo[11] 
                po[l11010] += vo[12] 
                po[l00110] += vo[13] 
                po[l10110] += vo[14] 
                po[l01110] += vo[15] 
                po[l11110] += vo[16]
                po[l00001] += vo[17] 
                po[l10001] += vo[18] 
                po[l01001] += vo[19] 
                po[l11001] += vo[20] 
                po[l00101] += vo[21] 
                po[l10101] += vo[22] 
                po[l01101] += vo[23] 
                po[l11101] += vo[24]
                po[l00011] += vo[25] 
                po[l10011] += vo[26] 
                po[l01011] += vo[27] 
                po[l11011] += vo[28] 
                po[l00111] += vo[29] 
                po[l10111] += vo[30] 
                po[l01111] += vo[31] 
                po[l11111] += vo[32]
            end
        end
    end
    total_itr = div(L, 32)
    parallel_run(total_itr, Threads.nthreads(), f, pos1, pos2, pos3, pos4, pos5, mask0, mask1, mask2, mask3, mask4, mask5, U, v, vout)
end

_apply_gate_threaded2!(key::Tuple{Int, Int, Int, Int, Int}, U::AbstractMatrix, v::AbstractVector, vout::AbstractVector) = _apply_fivebody_term_impl!(
	key, SMatrix{32,32, eltype(v)}(U), v, vout)



