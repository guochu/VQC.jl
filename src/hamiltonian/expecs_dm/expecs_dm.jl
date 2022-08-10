
include("expec_dm_serial.jl")
include("apply_qterms_dm.jl")



function expectation(m::QubitsTerm, state::DensityMatrix) 
	isempty(m) && return tr(state)
	if length(positions(m)) <= 3
	    return expectation_value_serial(Tuple(positions(m)), _get_mat(m), storage(state))
	else
		return tr(m * state)
	end
end


function expectation(m::QubitsOperator, state::DensityMatrix)
	if _largest_nterm(m) <= 3
		r = zero(eltype(state))
		for (k, v) in m.data
			r += expectation_value_serial(k, _get_mat(length(k), v), storage(state))
		end 
		return r   
	else
		return tr(m * state)
	end
end
