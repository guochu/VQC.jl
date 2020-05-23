@adjoint vdot(x::AbstractArray, y::AbstractArray) = vdot(x, y), z -> (conj(y) * z, conj(x) * z)
@adjoint cdot(x::AbstractArray, y::AbstractArray) = cdot(x, y), z -> (y * conj(z), x * z)

function gate_expec(state_a::AbstractVector, gate::AbstractGate, state_b::AbstractVector)
    T = promote_type(scalar_type(state_a), scalar_type(gate), scalar_type(state_b))
    state_b_t = convert(promote_type(typeof(state_b), T), copy(state_b))
    apply_gate!(gate, state_b_t)
    return vdot(state_a, state_b_t)
end

recursive_reverse(s::Number) = s

recursive_reverse(s::Nothing) = s
recursive_reverse(s::Vector{<:Number}) = reverse(s)
recursive_reverse(a::Vector) = [recursive_reverse(item) for item in reverse(a)]


function backward_evolution(result::AbstractVector, circuit::AbstractCircuit, z::AbstractVector)
    rc = circuit'
    r1 = []
    mps_tmp = copy(result)
    zt = copy(z)
    for item in rc
        isa(item, AbstractGate) || error("circuit must only contain pure gates.")
        apply_gate!(item, mps_tmp)
        df = differentiate(item')
        if isa(df, AbstractGate)
            push!(r1, gate_expec(zt, df, mps_tmp))
        else
            (df === nothing) || error("wrong circuit gradient.")
        end
        apply_gate!(conj(item), zt)
    end
    return mps_tmp, recursive_reverse(real([r1...])), zt
end


@adjoint *(circuit::AbstractCircuit, mps::AbstractVector) = begin
    result = circuit * mps
    return result, z -> begin
        r, grad, zt = backward_evolution(result, circuit, Vector{scalar_type(result)}(conj(z) ))
        return grad, conj(zt)
    end
end

@adjoint qstate(::Type{T}, mpsstr::Vector{<:Real}) where {T<:Number} = begin
    result = qstate(T, mpsstr)
    return result, z -> begin
        circuit = QCircuit([RyGate(i, Variable(theta*pi)) for (i, theta) in enumerate(mpsstr)])
        r, grad, zt = backward_evolution(result, circuit, Vector{scalar_type(result)}(conj(z) ))
        return nothing, grad .* pi
    end
end
