
function _expectation_value_util(pos::Int, m::AbstractMatrix, state::AbstractVector)
	(size(m, 1) == size(m, 2)) || error("observable must be a square matrix.")
	(size(m, 1) == 2) || error("input must be a 2 by 2 matrix.")
	sj = 2
	s1 = 2^(pos-1)
	s2 = s1 * sj
	r = zero(eltype(state))
	for i in 0:s2:(length(state)-1)
		for j in 0:(s1-1)
		    pos_start = i + j + 1
		    v = SVector(state[pos_start], state[pos_start + s1])
		    @fastmath r += dot(v, m, v)
		end
	end
	return r
end

function _expectation_value_util(pos::Int, m::AbstractMatrix, state::AbstractVector, state_c::AbstractVector)
	(size(m, 1) == size(m, 2)) || error("observable must be a square matrix.")
	(size(m, 1) == 2) || error("input must be a 2 by 2 matrix.")
	sj = 2
	s1 = 2^(pos-1)
	s2 = s1 * sj
	r = zero(eltype(state))
	for i in 0:s2:(length(state)-1)
		for j in 0:(s1-1)
		    pos_start = i + j + 1
		    v = SVector(state[pos_start], state[pos_start + s1])
		    v_c = SVector(state_c[pos_start], state_c[pos_start + s1])
		    @fastmath r += dot(v_c, m, v)
		end
	end
	return r
end

function _expectation_value_util(key::Tuple{Int, Int}, m::AbstractMatrix, v::AbstractVector)
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
                vi = SVector(v[l], v[l + pos1], v[l + pos2], v[l + pos1 + pos2])
                @fastmath r += dot(vi, m, vi)
            end
        end
    end
    return r
end


function _expectation_value_util(key::Tuple{Int, Int}, m::AbstractMatrix, v::AbstractVector, v_c::AbstractVector)
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
                vi = SVector(v[l], v[l + pos1], v[l + pos2], v[l + pos1 + pos2])
                vi_c = SVector(v_c[l], v_c[l + pos1], v_c[l + pos2], v_c[l + pos1 + pos2])
                @fastmath r += dot(vi_c, m, vi)
            end
        end
    end
    return r
end


function _expectation_value_util(key::Tuple{Int, Int, Int}, m::AbstractMatrix, v::AbstractVector, v_c::AbstractVector)
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
                	vi = SVector(v[l000], v[l100], v[l010], v[l110], v[l001], v[l101], v[l011], v[l111])
                	vi_c = SVector(v_c[l000], v_c[l100], v_c[l010], v_c[l110], v_c[l001], v_c[l101], v_c[l011], v_c[l111])
                	@fastmath r += dot(vi_c, m, vi)
            	end
        	end
    	end
    end
    return r
end

expectation_value_serial(pos::Int, m::AbstractMatrix, state::AbstractVector, state_c::AbstractVector) = _expectation_value_util(
	pos, SMatrix{2,2, eltype(state)}(m), state, state_c)
expectation_value_serial(pos::Int, m::AbstractMatrix, state::AbstractVector) = _expectation_value_util(
	pos, SMatrix{2,2, eltype(state)}(m), state)


expectation_value_serial(pos::Tuple{Int}, m::AbstractMatrix, state::AbstractVector, state_c::AbstractVector) = expectation_value_serial(
	pos[1], m, state, state_c)
expectation_value_serial(pos::Tuple{Int}, m::AbstractMatrix, state::AbstractVector) = expectation_value_serial(
	pos[1], m, state)

expectation_value_serial(pos::Tuple{Int, Int}, m::AbstractMatrix, state::AbstractVector, state_c::AbstractVector) = _expectation_value_util(
	pos, SMatrix{4,4, eltype(state)}(m), state, state_c)
expectation_value_serial(pos::Tuple{Int, Int}, m::AbstractMatrix, state::AbstractVector) = _expectation_value_util(
	pos, SMatrix{4,4, eltype(state)}(m), state)

expectation_value_serial(pos::Tuple{Int, Int, Int}, m::AbstractMatrix, state::AbstractVector, state_c::AbstractVector) = _expectation_value_util(
	pos, m, state, state_c) 
expectation_value_serial(pos::Tuple{Int, Int, Int}, m::AbstractMatrix, state::AbstractVector) = expectation_value_serial(
	pos, m, state, state)

