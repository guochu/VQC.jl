
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

function read_digits()
	data = JSON.parsefile("/Users/guochu/Documents/QuantumSimulator/Meteor.jl/project/seq2seq/digits/digits.txt")
	x, y = data["x"], data["y"]
	x = [[item...] for item in x]
	y = [y...]
	return x, y
end

function processing_data(x, y, p::Real)
    x = [reshape(item, 8, 8) for item in x]
    y = [Flux.onehot(item, 0:9) for item in y]
    L = length(y)
    Ntrain = round(Int, L * p)
    Ntest = L - Ntrain
    println("number of training $Ntrain, number of testing $Ntest.")
    println("start training...")
    return x[1:Ntrain], y[1:Ntrain], x[Ntrain+1:end], y[Ntrain+1:end]
end

function prepare_digits_data(p::Real)
    x, y = read_digits()
    return processing_data(x, y, p)
end

# x, y = read_digits()
#
# p = 0.8
#
# x_train,  y_train, x_test, y_test = prepare_data(x, y, p)
# # parameters
# # s1 = window_size_1
# # s2 = window_size_2
# depth = 4
# learn_rate = 0.05

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

function train_single(depth::Int=9, learn_rate::Real=0.01)
    p = 0.8
    x_train, y_train, x_test, y_test = prepare_digits_data(p)

    crs1 = [real_variational_circuit_1d(3*3, depth) for i in 1:2]
    circuit1 = QCNNLayer(crs1, (3, 3), padding=0)
    crs2 = [real_variational_circuit_1d(3*3, depth) for i in 1:2]
    circuit2 = QCNNLayer(crs2, (3, 3), padding=0)
    m = randn(10, 4 * length(crs1) * length(crs2))
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

    predict(input::Array{<:Real, 2}) = begin
        r = conv_net(input, circuit1, circuit2, m)
        return argmax(r)
    end

    accuracy(x, y) = begin
        r = [predict(x[i])== argmax(y[i]) for i in 1:length(x)]
        return sum(r)/length(r)
    end

    println("score before training $(accuracy(x_test, y_test)).")

    opt = ADAM(learn_rate)

    results = []
    los = []
    function train(epochs)
        for i in 1:epochs
            @time grad = parameters(gradient(loss, circuit1, circuit2, m))
#           paras = parameters(circuit)
            Optimise.update!(opt, x0, grad)
            set_parameters!(x0, circuit1, circuit2, m)
            if i % 1 == 0
                ac = accuracy(x_test, y_test)
                ss = loss(circuit1, circuit2, m)
                push!(results, ac)
                push!(los, ss)
                println("accuracy at the $i-th step is $ac, loss is $(ss).")
#               println("loss at step $i is $(loss(circuit1, circuit2, m))")
            end
        end
        return parameters(circuit1, circuit2, m)
    end

    epochs = 1000

    train(epochs)

    return results, los
end

train_single()

# nsamples = 10
#
# for i in 1:nsamples
#     println("the $i-th sample...")
#     results, los = train_single()
#     r = JSON.json(Dict("accuracy"=>results, "loss"=>los))
#     open("result/label2d$(depth)rate$(learn_rate)index$(i).txt", "w") do io
#         write(io, r)
#     end
# end








# # total number of parameters
# crs1 = [real_variational_circuit_1d(4, depth) for i in 1:2]
# circuit1 = QCNNLayer(crs1, (2, 2), padding=0)
# crs2 = [real_variational_circuit_1d(4, depth) for i in 1:1]
# circuit2 = QCNNLayer(crs2, (2, 2), padding=0)
# m = randn(length(crs1) * length(crs2))
#
# # total number of parameters
# x0 = parameters(circuit1, circuit2, m)
# # println(x0)
# # println(length(parameters(circuit1)))
# # println(length(parameters(circuit2)))
# # println(length(parameters(m)))
# println("total number of parameters $(length(x0))")
#
# loss(c1, c2, ma) = begin
#     r = 0.
#     for i in 1:length(x_train)
#         r += (conv_net(x_train[i], c1, c2, ma) - y_train[i])^2 / 2
#     end
#     return r
# end
#
# @time r = loss(circuit1, circuit2, m)
# println("initial loss is $r")
#
# # println("check if the gradient correct? $(check_gradient(loss, circuit1, circuit2, m))")
#
# predict(input::Array{<:Real, 2}) = begin
#     r = conv_net(input, circuit1, circuit2, m)
#     return r >= 0.5 ? 1 : 0
# end
#
# accuracy(x, y) = begin
#     r = [predict(x[i])==y[i] for i in 1:length(x)]
#     return sum(r)/length(r)
# end
#
# # distance(a, b) = dot(a, a) + dot(b, b) - 2*dot(a, b)
# # y_pred = [predict(item) for item in x_test]
# # score = sum([distance(a, b) for (a, b) in zip(y_pred, y_test)])
#
# println("score before training $(accuracy(x_test, y_test)).")
#
# opt = ADAM(learn_rate)
# # opt = Descent(learn_rate)
#
# results = []
# los = []
# function train(epochs)
#     for i in 1:epochs
#         @time grad = parameters(gradient(loss, circuit1, circuit2, m))
#         Optimise.update!(opt, x0, grad)
#         set_parameters!(x0, circuit1, circuit2, m)
#         if i % 1 == 0
#             ac = accuracy(x_test, y_test)
#             ss = loss(circuit1, circuit2, m)
#             push!(results, ac)
#             push!(results, ss)
#             println("accuracy at the $i-th step is $ac, loss is $(ss).")
# #             println("loss at step $i is $(loss(circuit1, circuit2, m))")
#         end
#     end
#     return parameters(circuit1, circuit2, m)
# end
#
# epochs = 1000
#
# train(epochs)
#
# r = JSON.json(Dict("accuracy"=>results, "loss"=>los))
# open("resultd$(depth)rate$(learn_rate).txt", "w") do io
#     write(io, r)
# end
