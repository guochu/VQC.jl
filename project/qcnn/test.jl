push!(LOAD_PATH, "../../src")

include("qcnn.jl")


using LinearAlgebra: norm, dot

using JSON
using Random: shuffle!

using Zygote

using Zygote: @adjoint

using Optim

using VQC

using Flux
using Flux.Optimise
using BenchmarkTools

max_pooling(a::Array{Float64, 3}) = begin
    b = maximum(a, dims=[1,2])
    return reshape(b, length(b))
end

function max_pooling_impl(m::Array{Float64, 3}, filter_shape::Tuple{Int, Int})
    n1, n2, n3 = size(m)
    s1, s2 = filter_shape
    (s1 <= n1 && s2 <= n2) || error("filter size too large.")
    out = Array{Float64, 3}(undef, n1-s1+1, n2-s2+1, n3)
    for i in 1:(n1-s1+1)
        for j in 1:(n2-s2+1)
            for k in 1:n3
                out[i, j, k] = maximum(m[i:(i+s1-1), j:(j+s2-1), k])
            end
        end
    end
    return out
end

@adjoint max_pooling_impl(m::Array{Float64, 3}, filter_shape::Tuple{Int, Int}) = begin
    n1, n2, n3 = size(m)
    s1, s2 = filter_shape
    (s1 <= n1 && s2 <= n2) || error("filter size too large.")
    out = Array{Float64, 3}(undef, n1-s1+1, n2-s2+1, n3)
    dout = Array{Any, 3}(undef, n1-s1+1, n2-s2+1, n3)
    for i in 1:(n1-s1+1)
        for j in 1:(n2-s2+1)
            for k in 1:n3
                # state_a = qstate(reshape(m[i:(i+s1-1), j:(j+s2-1), k], s1*s2))
                # tmp, dtmp =   dot(state_b, circuit * state_a)
                tmp, dtmp = Zygote.pullback(maximum, m[i:(i+s1-1), j:(j+s2-1), k])
                out[i, j, k] = tmp
                dout[i, j, k] = dtmp
            end
        end
    end
    return out, z -> begin
        dm = zeros(size(m))
        for i in 1:(n1-s1+1)
            for j in 1:(n2-s2+1)
                for k in 1:n3
                    a = dout[i, j, k](z[i, j, k])
                    dm[i:(i+s1-1), j:(j+s2-1), k] .+= a[1]
                end
            end
        end
        return dm, nothing
    end
end

max_pooling(m::Array{Float64, 3}, filter_shape::Tuple{Int, Int}, padding::Int=0) = max_pooling_impl(
    add_padding(m, padding), filter_shape)


function relu(s::Real)
    s < 0. ? 0. : s
end

@adjoint relu(s::Real) = relu(s), z -> s < 0. ? (0.,) : (z,)

function softmax(a::AbstractVector)
    b = exp.(a)
    return b ./ sum(b)
end


function conv_net(input::Array{<:Real, 2}, circuit1, circuit2, m)   # first quantum conv layer
    out1 = circuit1 * input
    # convert into 0 and 1s
    # println(size(out1))

    # second quantum conv layer
    out2 = circuit2 * out1
    # println(size(out2))

    # # maximum pooling
    # v = max_pooling(relu.(out2))
    out3 = max_pooling(out2, (3,3), 0)
    # println(size(out3))
#     # fully connected layer
#     v = m * v

#     # softmax
#     v = softmax(v)
    return softmax(m * reshape(out3, length(out3)))
end

distance(x::AbstractVector, y::AbstractVector) = dot(x, x) + dot(y, y) - 2 * dot(x, y)

function train_single(depth::Int=4, learn_rate::Real=0.01)
    p = 0.8
    x_train = [rand(8, 8) for i in 1:100]
    y_train = [rand(2) for i in 1:100]

    crs1 = [real_variational_circuit_1d(3*3, depth) for i in 1:2]
    circuit1 = QCNNLayer(crs1, (3, 3), padding=0)
    crs2 = [real_variational_circuit_1d(3*3, depth) for i in 1:3]
    circuit2 = QCNNLayer(crs2, (3, 3), padding=0)
    m = randn(2, 4 * length(crs1) * length(crs2))
    x0 = parameters(circuit1, circuit2, m)
    println("total number of parameters $(length(x0)).")
    println("number of parameters in the last layer $(length(m)).")
    loss(c1, c2, ma) = begin
        r = 0.
        for i in 1:length(x_train)
            r += distance(conv_net(x_train[i], c1, c2, ma), y_train[i])
        end
        return r / length(x_train)
    end

    @time r = loss(circuit1, circuit2, m)
    println("initial loss is $r")

    grad = gradient(loss, circuit1, circuit2, m)
    @time grad = gradient(loss, circuit1, circuit2, m)

    # println(check_gradient(loss, circuit1, circuit2, m, verbose=1))
end

train_single()
