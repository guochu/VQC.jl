
@adjoint *(circuit::QCircuit, x::Union{StateVector, DensityMatrix}) = begin
    y = circuit * x
    return y, Δ -> begin
        Δ, grads, y = back_propagate(copy(Δ), circuit, copy(y))
        return grads, Δ
    end
end

@adjoint qubit_encoding(::Type{T}, mpsstr::Vector{<:Real}) where {T<:Number} = begin
    y = qubit_encoding(T, mpsstr)
    return y, Δ -> begin
        circuit = QCircuit([RyGate(i, theta*pi, isparas=true) for (i, theta) in enumerate(mpsstr)])
        Δ, grads, y = back_propagate(Δ, circuit, copy(y))
        return nothing, grads .* pi
    end
end

# @adjoint amplitude_encoding(::Type{T}, v::AbstractVector{<:Number}; kwargs...) where {T <: Number} = amplitude_encoding(
#     T, v; kwargs...), z -> (nothing, z[1:length(v)])


function back_propagate(Δ::AbstractVector, m::Gate, y::StateVector)
    Δ = StateVector(Δ, nqubits(y))
    Δ = apply!(m', Δ)
    y = apply!(m', y)
    ∇θs = nothing
    if nparameters(m) > 0
        ∇θs = [real(expectation(y, item, Δ)) for item in differentiate(m)]
    end
    return storage(Δ), ∇θs, y
end

# a temporary solution, which requires an additional copy of the density matrix
# we can not assume x or y to be positive or hermitian here, since they may not be physical density matrix
function expectation(x::DensityMatrix, m::Gate, y::DensityMatrix)
    yc = copy(y)
    apply_threaded!(m, yc.data)
    return dot(storage(x)', storage(yc))
end 

function back_propagate(Δ::AbstractMatrix, m::Gate, y::DensityMatrix)
    Δ = DensityMatrix(Δ, nqubits(y))
    Δ = apply!(m', Δ)
    y = apply!(m', y)
    ∇θs = nothing
    if nparameters(m) > 0
        ∇θs = [real(expectation(y, item, Δ) + expectation(Δ, item', y)) for item in differentiate(m)]
    end
    return storage(Δ), ∇θs, y
end

function back_propagate(Δ::AbstractMatrix, m::QuantumMap, y::DensityMatrix)
    Δ = DensityMatrix(Δ, nqubits(y))
    Δ = apply_dagger!(m, Δ)
    y = apply_inverse!(m, y)
    ∇θs = nothing
    return storage(Δ), ∇θs, y
end

function back_propagate_util(Δ, circuit::QCircuit, y)
    RT = real(eltype(y))
    grads = Vector{RT}[]
    for item in reverse(circuit)
        Δ, ∇θs, y = back_propagate(Δ, item, y)
        !isnothing(∇θs) && push!(grads, ∇θs)
    end

    ∇θs_all = RT[]
    for item in Iterators.reverse(grads)
        append!(∇θs_all, item)
    end

    return Δ, ∇θs_all, y
end

back_propagate(Δ::AbstractVector, circuit::QCircuit, y::StateVector) = back_propagate_util(Δ, circuit, y)
back_propagate(Δ::AbstractMatrix, circuit::QCircuit, y::DensityMatrix) = back_propagate_util(Δ, circuit, y)
 
 


# function back_propagate(Δ::AbstractMatrix, m::Gate, y::DensityMatrix)
#     Δ = StateVector(Δ, nqubits(y))
#     Δ = apply!(m', Δ)
#     y = apply!(m', y)
#     ∇θs = nothing
#     if nparameters(m) > 0
#         ∇θs = [real(expectation(y, item, Δ)) for item in differentiate(m)]
#     end
#     return storage(Δ), ∇θs, y
# end

