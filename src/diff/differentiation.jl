export differentiate, nparameters

"""
Differentiate gates
"""

differentiate(s::AbstractQuantumOperation) = error("differentiate not implemented for operation type $(typeof(s)).")
differentiate(s::AbstractGate) = nothing
nparameters(s::AbstractGate) = 0

set_parameters_impl!(s::AbstractGate, coeff::AbstractVector{<:Number}, start_pos::Int=1) = start_pos
collect_variables_impl!(a::Vector, b::AbstractGate) = nothing

function differentiate(s::AdjointGate)
	r = differentiate(inner_gate(s))
	if r === nothing
	    return r
	else
		return AdjointGate(r)
	end
end

function differentiate(s::TransposeGate) 
	r = differentiate(inner_gate(s))
	if r === nothing
	    return r
	else
		return TransposeGate(r)
	end
end

function differentiate(s::ConjugateGate)
	r = differentiate(inner_gate(s))
	if r === nothing
	    return r
	else
		return ConjugateGate(r)
	end
end 

differentiate(s::RxGate{<:Variable}) =  RxGate(key(s), s.parameter + 0.5*pi)
nparameters(s::RxGate{<:Variable}) = 1
set_parameters_impl!(s::RxGate{<:Variable}, coeff::AbstractVector{<:Number}, start_pos::Int=1) = set_parameters_impl!(
	s.parameter, coeff, start_pos)
collect_variables_impl!(a::Vector, b::RxGate{<:Variable}) = push!(a, value(b.parameter))

differentiate(s::RyGate{<:Variable}) = RyGate(key(s), s.parameter + 0.5*pi)
nparameters(s::RyGate{<:Variable}) = 1
set_parameters_impl!(s::RyGate{<:Variable}, coeff::AbstractVector{<:Number}, start_pos::Int=1) = set_parameters_impl!(
	s.parameter, coeff, start_pos)
collect_variables_impl!(a::Vector, b::RyGate{<:Variable}) = push!(a, value(b.parameter))

differentiate(s::RzGate{<:Variable}) = RzGate(key(s), s.parameter + 0.5*pi)
nparameters(s::RzGate{<:Variable}) = 1
set_parameters_impl!(s::RzGate{<:Variable}, coeff::AbstractVector{<:Number}, start_pos::Int=1) = set_parameters_impl!(
	s.parameter, coeff, start_pos)
collect_variables_impl!(a::Vector, b::RzGate{<:Variable}) = push!(a, value(b.parameter))


# controled rotation gates
differentiate(s::CRxGate{<:Variable}) = begin
    m = permute(reshape(_row_kron(DOWN, Rx(value(s.parameter+0.5*pi))), 2,2,2,2), s.perm)
    return TwoBodyGate(key(s), m)
end 
set_parameters_impl!(s::CRxGate{<:Variable}, coeff::AbstractVector{<:Number}, start_pos::Int=1) = set_parameters_impl!(
	s.parameter, coeff, start_pos)
collect_variables_impl!(a::Vector, b::CRxGate{<:Variable}) = push!(a, value(b.parameter))
nparameters(s::CRxGate{<:Variable}) = 1

differentiate(s::CRyGate{<:Variable}) = begin
    m = permute(reshape(_row_kron(DOWN, Ry(value(s.parameter+0.5*pi))), 2,2,2,2), s.perm)
    return TwoBodyGate(key(s), m)
end 
set_parameters_impl!(s::CRyGate{<:Variable}, coeff::AbstractVector{<:Number}, start_pos::Int=1) = set_parameters_impl!(
	s.parameter, coeff, start_pos)
collect_variables_impl!(a::Vector, b::CRyGate{<:Variable}) = push!(a, value(b.parameter))
nparameters(s::CRyGate{<:Variable}) = 1

differentiate(s::CRzGate{<:Variable}) = begin
    m = permute(reshape(_row_kron(DOWN, Rz(value(s.parameter+0.5*pi))), 2,2,2,2), s.perm)
    return TwoBodyGate(key(s), m)
end 
set_parameters_impl!(s::CRzGate{<:Variable}, coeff::AbstractVector{<:Number}, start_pos::Int=1) = set_parameters_impl!(
	s.parameter, coeff, start_pos)
collect_variables_impl!(a::Vector, b::CRzGate{<:Variable}) = push!(a, value(b.parameter))
nparameters(s::CRzGate{<:Variable}) = 1




nparameters(s::AbstractCircuit) = isempty(s) ? 0 : sum([nparameters(gate) for gate in s])

function collect_variables_impl!(r::Vector, s::AbstractCircuit)
	for gate in s
	    collect_variables_impl!(r, gate)
	end
end

function set_parameters_impl!(s::AbstractCircuit, coeff::AbstractVector{<:Number}, start_pos::Int=1) 
	for item in s
		start_pos = set_parameters_impl!(item, coeff, start_pos)
	end
	return start_pos
end

"""
Differentiate circuit
"""
function differentiate(x::AbstractCircuit)
	r = []
	for i in 1:length(x)
	    df = differentiate(x[i])
	    if df === nothing
	        continue
	    end
	    if isa(df, Vector)
	        for item in df
	        	isa(item, AbstractCircuit) || error("wrong type...")
	            push!(r, QCircuit([[x[l] for l in 1:(i-1)]..., item, [x[l] for l in (i+1):length(x)]...]))
	        end
	    else
	    	push!(r, QCircuit([[x[l] for l in 1:(i-1)]..., df, [x[l] for l in (i+1):length(x)]...]))
	    end
	end
	return r
end
