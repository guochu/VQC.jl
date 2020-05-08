
function _apply_gate_2_impl!(key::Int, U::AbstractMatrix, v::AbstractVector)
    L = length(v)
    pos = 2^(key-1)
    stride = pos * 2
    for i in 0:stride:(L-1)
        @inbounds for j in 0:(pos-1)
            l = i + j + 1
            vi1 = v[l]
            vi2 = v[l + pos]
            vo1 = U[1,1] * vi1 + U[1,2] * vi2
            vo2 = U[2,1] * vi1 + U[2,2] * vi2
            v[l] = vo1
            v[l + pos] = vo2
        end
    end
end

_apply_gate_2!(key::Int, U::AbstractMatrix, v::AbstractVector{<:Real}) = _apply_gate_2_impl!(
key, SMatrix{2,2, eltype(v)}(U), v)
_apply_gate_2!(key::Int, U::AbstractMatrix{<:Complex}, v::AbstractVector{Complex{T}}) where {T<:Real} = _apply_gate_2_impl!(
key, SMatrix{2,2,eltype(v)}(U), v)
_apply_gate_2!(key::Int, U::AbstractMatrix{<:Real}, v::AbstractVector{Complex{T}}) where {T<:Real} = _apply_gate_2_impl!(
key, SMatrix{2,2, T}(U), v)


function _apply_gate_2_impl!(key::Tuple{Int, Int}, U::AbstractMatrix, v::AbstractVector)
    # U = Matrix(transpose(reshape(U, 4, 4)))
    L = length(v)
    q1, q2 = key
    pos1, pos2 = 2^(q1-1), 2^(q2-1)
    stride1, stride2 = 2 * pos1, 2 * pos2
    for i in 0:stride2:(L-1)
        for j in 0:stride1:(pos2-1)
            @inbounds for k in 0:(pos1-1)
                l = i + j + k + 1
                # vi1 = v[l]
                # vi2 = v[l + pos1]
                # vi3 = v[l + pos2]
                # vi4 = v[l + pos1 + pos2]
                # # transposed
                # vo1 = U[1,1] * vi1 + U[2,1] * vi2 + U[3,1] * vi3 + U[4,1] * vi4
                # vo2 = U[1,2] * vi1 + U[2,2] * vi2 + U[3,2] * vi3 + U[4,2] * vi4
                # vo3 = U[1,3] * vi1 + U[2,3] * vi2 + U[3,3] * vi3 + U[4,3] * vi4
                # vo4 = U[1,4] * vi1 + U[2,4] * vi2 + U[3,4] * vi3 + U[4,4] * vi4
                # v[l] = vo1
                # v[l + pos1] = vo2
                # v[l + pos2] = vo3
                # v[l + pos1 + pos2] = vo4
                vi = SVector(v[l], v[l + pos1], v[l + pos2], v[l + pos1 + pos2])
                v[l], v[l + pos1], v[l + pos2], v[l + pos1 + pos2] = U * vi
                # v[l] = vo[1]
                # v[l + pos1] = vo[2]
                # v[l + pos2] = vo[3]
                # v[l + pos1 + pos2] = vo[4]
            end
        end
    end
end

_apply_gate_2!(key::Tuple{Int, Int}, U::AbstractMatrix, v::AbstractVector{<:Real}) =  _apply_gate_2_impl!(
key, SMatrix{4,4, eltype(v)}(U), v)
_apply_gate_2!(key::Tuple{Int, Int}, U::AbstractMatrix{<:Complex}, v::AbstractVector{Complex{T}}) where {T<:Real} = _apply_gate_2_impl!(
key, SMatrix{4,4,eltype(v)}(U), v)
_apply_gate_2!(key::Tuple{Int, Int}, U::AbstractMatrix{<:Real}, v::AbstractVector{Complex{T}}) where {T<:Real} = _apply_gate_2_impl!(
key, SMatrix{4,4, T}(U), v)
_apply_gate_2!(key::Tuple{Int, Int}, U::AbstractArray{T, 4}, v::AbstractVector) where T = _apply_gate_2!(key, reshape(U, 4, 4), v)


_apply_gate_2!(key::Tuple{Int, Int, Int}, U::AbstractMatrix{T}, v::AbstractVector{T}) where T = _apply_gate_impl(
key, U, v)
apply_gate!(x::AbstractGate, s::Vector) = _apply_gate_2!(key(x), op(x), s)
