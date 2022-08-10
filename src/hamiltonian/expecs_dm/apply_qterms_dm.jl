


# apply hamiltonian term on density matrix


function (m::QubitsTerm)(vr::DensityMatrix)
	v = vr.data
	vout = similar(v)
	_apply_qterm_util!(m, v, vout)
	return DensityMatrix(vout, nqubits(vr))
end

function (m::QubitsOperator)(vr::DensityMatrix) 
	v = vr.data
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
	return DensityMatrix(vout, nqubits(vr))
end


Base.:*(m::QubitsOperator, v::DensityMatrix) = m(v)
Base.:*(m::QubitsTerm, v::DensityMatrix) = m(v)
