

include("expec_serial.jl")
include("expec_threaded.jl")
include("expec_threaded_2.jl")

include("expec_high_qubit_terms.jl")
include("expec_high_qubit_terms_2.jl")



function expectation(m::QubitsTerm, state::StateVector) 
	isempty(m) && return dot(state, state)
	if length(positions(m)) <= LARGEST_SUPPORTED_NTERMS
	    return expectation_value(Tuple(positions(m)), _get_mat(m), storage(state))
	else
		return dot(state, m * state)
	end
end


function expectation(m::QubitsOperator, state::StateVector)
	if _largest_nterm(m) <= LARGEST_SUPPORTED_NTERMS
		r = zero(eltype(state))
		for (k, v) in m.data
			r += expectation_value(k, _get_mat(length(k), v), storage(state))
		end 
		return r   
	else
		state = storage(state)
		workspace = similar(state)
		state_2 = zeros(eltype(state), length(state))
		for (k, v) in m.data
		    for item in v
		    	_apply_qterm_util!(QubitsTerm(k, item[1], item[2]), state, workspace)
		    	state_2 .+= workspace
		    end
		end
		return dot(state, state_2)
	end
end


function expectation(state_c::StateVector, m::QubitsTerm, state::StateVector) 
	isempty(m) && return dot(state_c, state)
	if length(positions(m)) <= LARGEST_SUPPORTED_NTERMS
		return expectation_value(Tuple(positions(m)), _get_mat(m), storage(state), storage(state_c))
	else
		return dot(state_c, m * state)
	end	
end

function expectation(state_c::StateVector, m::QubitsOperator, state::StateVector)
	if _largest_nterm(m) <= LARGEST_SUPPORTED_NTERMS
		r = zero(eltype(state))
		for (k, v) in m.data
			r += expectation_value(k, _get_mat(length(k), v), storage(state), storage(state_c))
		end
		return r
	else
		state = storage(state)
		workspace = similar(state)
		state_2 = zeros(eltype(state), length(state))
		for (k, v) in m.data
		    for item in v
		    	_apply_qterm_util!(QubitsTerm(k, item[1], item[2]), state, workspace)
		    	state_2 .+= workspace
		    end
		end
		return dot(state_c, state_2)
	end
end

expectation(m::Gate, state::StateVector) = expectation_value(ordered_positions(m), ordered_mat(m), storage(state))
expectation(state_c::StateVector, m::Gate, state::StateVector) = expectation_value(
	ordered_positions(m), ordered_mat(m), storage(state), storage(state_c))

expectation(state_c::StateVector, m::AbstractMatrix, state::StateVector) = dot(storage(state_c), m, storage(state))
expectation(m::AbstractMatrix, state::StateVector) = dot(state, m, state)



expectation_value(pos::Int, m::AbstractMatrix, state::AbstractVector) = (length(state) >= 32) ? expectation_value_threaded(
	pos, m, state) : expectation_value_serial(pos, m, state)
expectation_value(pos::Int, m::AbstractMatrix, state::AbstractVector, state_c::AbstractVector) = (length(state) >= 32) ? expectation_value_threaded(
	pos, m, state, state_c) : expectation_value_serial(pos, m, state, state_c)
expectation_value(pos::Tuple{Int}, m::AbstractMatrix, state::AbstractVector) = expectation_value(pos[1], m, state)
expectation_value(pos::Tuple{Int}, m::AbstractMatrix, state::AbstractVector, state_c::AbstractVector) = expectation_value(
	pos[1], m, state, state_c)

expectation_value(pos::Tuple{Int, Int}, m::AbstractMatrix, state::AbstractVector) = (length(state) >= 32) ? expectation_value_threaded(
	pos, m, state) : expectation_value_serial(pos, m, state)
expectation_value(pos::Tuple{Int, Int}, m::AbstractMatrix, state::AbstractVector, state_c::AbstractVector) = (length(state) >= 32) ? expectation_value_threaded(
	pos, m, state, state_c) : expectation_value_serial(pos, m, state, state_c)
expectation_value(pos::Tuple{Int, Int}, m::AbstractArray{T, 4}, state::AbstractVector) where T = expectation_value(
	pos, reshape(m, 4, 4), state)
expectation_value(pos::Tuple{Int, Int}, m::AbstractArray{T, 4}, state::AbstractVector, state_c::AbstractVector) where T = expectation_value(
	pos, reshape(m, 4, 4), state, state_c)


expectation_value(pos::Tuple{Int, Int, Int}, m::AbstractMatrix, state::AbstractVector) = (length(state) >= 32) ? expectation_value_threaded(
	pos, m, state) : expectation_value_serial(pos, m, state)
expectation_value(pos::Tuple{Int, Int, Int}, m::AbstractMatrix, state::AbstractVector, state_c::AbstractVector) = (length(state) >= 32) ? expectation_value_threaded(
	pos, m, state, state_c) : expectation_value_serial(pos, m, state, state_c)
expectation_value(pos::Tuple{Int, Int, Int}, m::AbstractArray{T, 6}, state::AbstractVector) where T = expectation_value(
	pos, reshape(m, 8, 8), state)
expectation_value(pos::Tuple{Int, Int, Int}, m::AbstractArray{T, 6}, state::AbstractVector, state_c::AbstractVector) where T = expectation_value(
	pos, reshape(m, 8, 8), state, state_c)


expectation_value(pos::Tuple{Int, Int, Int, Int}, m::AbstractMatrix, state::AbstractVector) = expectation_value_threaded(
	pos, m, state) 
expectation_value(pos::Tuple{Int, Int, Int, Int}, m::AbstractMatrix, state::AbstractVector, state_c::AbstractVector) = expectation_value_threaded(
	pos, m, state, state_c) 
expectation_value(pos::Tuple{Int, Int, Int, Int}, m::AbstractArray{T, 8}, state::AbstractVector) where T = expectation_value(
	pos, Matrix{eltype(state)}(reshape(m, 16, 16)), state)
expectation_value(pos::Tuple{Int, Int, Int, Int}, m::AbstractArray{T, 8}, state::AbstractVector, state_c::AbstractVector) where T = expectation_value(
	pos, Matrix{eltype(state)}(reshape(m, 16, 16)), state, state_c)


expectation_value(pos::Tuple{Int, Int, Int, Int, Int}, m::AbstractMatrix, state::AbstractVector) = expectation_value_threaded(
	pos, m, state) 
expectation_value(pos::Tuple{Int, Int, Int, Int, Int}, m::AbstractMatrix, state::AbstractVector, state_c::AbstractVector) = expectation_value_threaded(
	pos, m, state, state_c) 
expectation_value(pos::Tuple{Int, Int, Int, Int, Int}, m::AbstractArray{T, 10}, state::AbstractVector) where T = expectation_value(
	pos, Matrix{eltype(state)}(reshape(m, 32, 32)), state)
expectation_value(pos::Tuple{Int, Int, Int, Int, Int}, m::AbstractArray{T, 10}, state::AbstractVector, state_c::AbstractVector) where T = expectation_value(
	pos, Matrix{eltype(state)}(reshape(m, 32, 32)), state, state_c)

