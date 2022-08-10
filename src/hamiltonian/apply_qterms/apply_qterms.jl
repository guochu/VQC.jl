include("short_range_serial.jl")
include("short_range_threaded.jl")
include("long_range_threaded.jl")


function (m::QubitsTerm)(vr::StateVector)
	v = storage(vr)
	vout = similar(v)
	_apply_qterm_util!(m, v, vout)
	return StateVector(vout, nqubits(vr))
end

function (m::QubitsOperator)(vr::StateVector) 
	v = storage(vr)
	vout = zeros(eltype(v), length(v))
	if _largest_nterm(m) <= LARGEST_SUPPORTED_NTERMS
		_apply_util!(m, v, vout)
	else
		workspace = similar(v)
		for (k, dd) in m.data
			for item in dd
			   _apply_qterm_util!(QubitsTerm(k, item[1], item[2]), v, workspace) 
			   vout .+= workspace
			end
		end
	end
	return StateVector(vout, nqubits(vr))
end


Base.:*(m::QubitsOperator, v::StateVector) = m(v)
Base.:*(m::QubitsTerm, v::StateVector) = m(v)



const LARGEST_SUPPORTED_NTERMS = 5


function _largest_nterm(x::QubitsOperator)
	n = 0
	for (k, v) in x.data
		n = max(n, length(k))
	end
	return n
end

function _apply_qterm_util!(m::QubitsTerm, v::AbstractVector, vout::AbstractVector)
	tmp = coeff(m)
	@. vout = tmp * v
	if length(v) >= 32
		for (pos, mat) in zip(positions(m), oplist(m))
			_apply_gate_threaded2!(pos, mat, vout)
		end	
	else    
		for (pos, mat) in zip(positions(m), oplist(m))
			_apply_gate_2!(pos, mat, vout)
		end			
	end
end


_apply_util!(m::QubitsOperator, v::AbstractVector, vout::AbstractVector) = (length(v) >= 32) ? _apply_threaded_util!(
    m, v, vout) : _apply_serial_util!(m, v, vout)

