
@inline get_2_by_2(state::AbstractMatrix, posa::Int, posb::Int) = SMatrix{2,2}(state[posa, posa], state[pos_b, posa], state[posa, pos_b], state[pos_b, pos_b])

@inline function get_4_by_4(state::AbstractMatrix, pos1::Int, pos2::Int, pos3::Int, pos4::Int)
	return SMatrix{4,4}(state[pos1, pos1], state[pos2, pos1], state[pos3, pos1], state[pos4, pos1], 
						state[pos1, pos2], state[pos2, pos2], state[pos3, pos2], state[pos4, pos2],
						state[pos1, pos3], state[pos2, pos3], state[pos3, pos3], state[pos4, pos3],
						state[pos1, pos4], state[pos2, pos4], state[pos3, pos4], state[pos4, pos4])
end 

@inline function get_8_by_8(state::AbstractMatrix, pos1::Int, pos2::Int, pos3::Int, pos4::Int, pos5::Int, pos6::Int, pos7::Int, pos8::Int)
	return SMatrix{8,8}(state[pos1, pos1], state[pos2, pos1], state[pos3, pos1], state[pos4, pos1], state[pos5, pos1], state[pos6, pos1], state[pos7, pos1], state[pos8, pos1],
						state[pos1, pos2], state[pos2, pos2], state[pos3, pos2], state[pos4, pos2], state[pos5, pos2], state[pos6, pos2], state[pos7, pos2], state[pos8, pos2],
						state[pos1, pos3], state[pos2, pos3], state[pos3, pos3], state[pos4, pos3], state[pos5, pos3], state[pos6, pos3], state[pos7, pos3], state[pos8, pos3],
						state[pos1, pos4], state[pos2, pos4], state[pos3, pos4], state[pos4, pos4], state[pos5, pos4], state[pos6, pos4], state[pos7, pos4], state[pos8, pos4],
						state[pos1, pos5], state[pos2, pos5], state[pos3, pos5], state[pos4, pos5], state[pos5, pos5], state[pos6, pos5], state[pos7, pos5], state[pos8, pos5],
						state[pos1, pos6], state[pos2, pos6], state[pos3, pos6], state[pos4, pos6], state[pos5, pos6], state[pos6, pos6], state[pos7, pos6], state[pos8, pos6],
						state[pos1, pos7], state[pos2, pos7], state[pos3, pos7], state[pos4, pos7], state[pos5, pos7], state[pos6, pos7], state[pos7, pos7], state[pos8, pos7],
						state[pos1, pos8], state[pos2, pos8], state[pos3, pos8], state[pos4, pos8], state[pos5, pos8], state[pos6, pos8], state[pos7, pos8], state[pos8, pos8])
end 


function _expectation_value_util(pos::Int, m::AbstractMatrix, state::AbstractMatrix)
	(size(m, 1) == size(m, 2)) || error("observable must be a square matrix.")
	(size(m, 1) == 2) || error("input must be a 2 by 2 matrix.")
	sj = 2
	s1 = 2^(pos-1)
	s2 = s1 * sj
	r = zero(eltype(state))
	for i in 0:s2:(length(state)-1)
		for j in 0:(s1-1)
		    v = get_2_by_2(state, pos_start, pos_start + s1)
		    @fastmath r += tr(m * v)
		end
	end
	return r
end

function _expectation_value_util(pos::Int, m::AbstractMatrix, state::AbstractMatrix, state_c::AbstractMatrix)
	(size(m, 1) == size(m, 2)) || error("observable must be a square matrix.")
	(size(m, 1) == 2) || error("input must be a 2 by 2 matrix.")
	sj = 2
	s1 = 2^(pos-1)
	s2 = s1 * sj
	r = zero(eltype(state))
	for i in 0:s2:(length(state)-1)
		for j in 0:(s1-1)
		    pos_start = i + j + 1
		    pos_b = pos_start + s1
		    v = get_2_by_2(state, pos_start, pos_b)
		    v_c = get_2_by_2(state_c, pos_start, pos_b)
		    @fastmath r += dot(v_c, m, v)
		end
	end
	return r
end

function _expectation_value_util(key::Tuple{Int, Int}, m::AbstractMatrix, v::AbstractMatrix)
	(size(m, 1) == size(m, 2)) || error("observable must be a square matrix.")
	(size(m, 1) == 4) || error("input must be a 4 by 4 matrix.")
    L = length(v)
    q1, q2 = key
    pos1, pos2 = 2^(q1-1), 2^(q2-1)
    stride1, stride2 = 2 * pos1, 2 * pos2
    r = zero(eltype(v))
    for i in 0:stride2:(L-1)
        for j in 0:stride1:(pos2-1)
            @inbounds for k in 0:(pos1-1)
                l = i + j + k + 1
                vi = get_4_by_4(v, l, l + pos1, l + pos2, l + pos1 + pos2)
                @fastmath r += tr(m * vi)
            end
        end
    end
    return r
end


function _expectation_value_util(key::Tuple{Int, Int}, m::AbstractMatrix, v::AbstractMatrix, v_c::AbstractMatrix)
	(size(m, 1) == size(m, 2)) || error("observable must be a square matrix.")
	(size(m, 1) == 4) || error("input must be a 4 by 4 matrix.")
    L = length(v)
    q1, q2 = key
    pos1, pos2 = 2^(q1-1), 2^(q2-1)
    stride1, stride2 = 2 * pos1, 2 * pos2
    r = zero(eltype(v))
    for i in 0:stride2:(L-1)
        for j in 0:stride1:(pos2-1)
            @inbounds for k in 0:(pos1-1)
                l = i + j + k + 1
                vi = get_4_by_4(v, l, l + pos1, l + pos2, l + pos1 + pos2)
                vi_c = get_4_by_4(v_c, l, l + pos1, l + pos2, l + pos1 + pos2)
                @fastmath r += dot(vi_c, m, vi)
            end
        end
    end
    return r
end


function _expectation_value_util(key::Tuple{Int, Int, Int}, m::AbstractMatrix, v::AbstractMatrix, v_c::AbstractMatrix)
    L = length(v)
    q1, q2, q3 = key
    pos1, pos2, pos3 = 2^(q1-1), 2^(q2-1), 2^(q3-1)
    stride1, stride2, stride3 = 2 * pos1, 2 * pos2, 2 * pos3
    r = zero(eltype(v))
   	for h in 0:stride3:(L-1)
   		for i in 0:stride2:(pos3-1)
   			for j in 0:stride1:(pos2-1)
   				@inbounds for k in 0:(pos1-1)
   					l000 = h + i + j + k + 1
   					l100 = l000 + pos1
            		l010 = l000 + pos2
            		l110 = l010 + pos1

            		l001 = l000 + pos3
            		l101 = l001 + pos1
            		l011 = l001 + pos2
            		l111 = l011 + pos1
                	vi = get_8_by_8(v, l000, l100, l010, l110, l001, l101, l011, l111)
                	vi_c = get_8_by_8(v_c, l000, l100, l010, l110, l001, l101, l011, l111)
                	@fastmath r += dot(vi_c, m, vi)
            	end
        	end
    	end
    end
    return r
end

expectation_value_serial(pos::Int, m::AbstractMatrix, state::AbstractMatrix, state_c::AbstractMatrix) = _expectation_value_util(
	pos, SMatrix{2,2, eltype(state)}(m), state, state_c)
expectation_value_serial(pos::Int, m::AbstractMatrix, state::AbstractMatrix) = _expectation_value_util(
	pos, SMatrix{2,2, eltype(state)}(m), state)


expectation_value_serial(pos::Tuple{Int}, m::AbstractMatrix, state::AbstractMatrix, state_c::AbstractMatrix) = expectation_value_serial(
	pos[1], m, state, state_c)
expectation_value_serial(pos::Tuple{Int}, m::AbstractMatrix, state::AbstractMatrix) = expectation_value_serial(
	pos[1], m, state)

expectation_value_serial(pos::Tuple{Int, Int}, m::AbstractMatrix, state::AbstractMatrix, state_c::AbstractMatrix) = _expectation_value_util(
	pos, SMatrix{4,4, eltype(state)}(m), state, state_c)
expectation_value_serial(pos::Tuple{Int, Int}, m::AbstractMatrix, state::AbstractMatrix) = _expectation_value_util(
	pos, SMatrix{4,4, eltype(state)}(m), state)

expectation_value_serial(pos::Tuple{Int, Int, Int}, m::AbstractMatrix, state::AbstractMatrix, state_c::AbstractMatrix) = _expectation_value_util(
	pos, m, state, state_c) 
expectation_value_serial(pos::Tuple{Int, Int, Int}, m::AbstractMatrix, state::AbstractMatrix) = expectation_value_serial(
	pos, m, state, state)

