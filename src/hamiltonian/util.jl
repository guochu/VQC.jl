

_get_mat(x::Tuple{Vector{AbstractMatrix}, Number}) = QuantumCircuits._kron_ops(reverse(x[1])) * x[2]
_get_mat(x::QubitsTerm) = QuantumCircuits._kron_ops(reverse(oplist(x))) * coeff(x)

function _get_mat(n::Int, x::QuantumCircuits.QOP_DATA_VALUE_TYPE)
    isempty(x) && error("bond is empty.")
    m = zeros(_scalar_type(x), 2^n, 2^n)
    for item in x
        tmp = QuantumCircuits._kron_ops(reverse(item[1]))
        alpha = item[2]
        @. m += alpha * tmp
    end
    return m
end


function _scalar_type(x::QuantumCircuits.QOP_DATA_VALUE_TYPE)
	T = Int
	for (k, coef) in x
		T = promote_type(T, typeof(coef))
		for item in k
			T = promote_type(T, eltype(item))
		end
	end
	return T
end

