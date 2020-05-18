
push!(LOAD_PATH, "../../src")

include("qcnn.jl")


using LinearAlgebra: norm, dot

using JSON
using Random: shuffle!

using Zygote

using Zygote: @adjoint

using Optim

using VQC

using Flux.Optimise


function prepare_data(path)
    data = JSON.parsefile(path)
    xtrain = data["x_train"]
    xtrain = [Array(reshape([vcat(item...)...], 3, 3)') for item in xtrain]
    xtest = data["x_test"]
    xtest = [Array(reshape([vcat(item...)...], 3, 3)') for item in xtest]
    ytrain = data["y_train"]
    ytest = data["y_test"]
    return xtrain, [ytrain...], xtest, [ytest...]
end

path = "data.json"

x_train,  y_train, x_test, y_test = prepare_data(path)
# parameters
# s1 = window_size_1
# s2 = window_size_2
depth = 4
learn_rate = 0.05
# n_filter = 2

ntrain = length(y_train)
ntest = length(y_test)

println("number of training $ntrain, number of testing $ntest.")
println("start training...")

max_pooling(a::Array{Float64, 3}) = begin
    b = maximum(a, dims=[1,2])
    return reshape(b, length(b))
end

# function softmax(a::AbstractVector)
#     b = exp.(a)
#     return b ./ sum(b)
# end

# function one_hot(i, d)
#     i = convert(Int, i+1)
#     d = convert(Int, d)
#     (i >= 1 && i <= d) || error("out of range.")
#     r = zeros(Float64, d)
#     r[i] = 1
#     return r
# end

function relu(s::Real)
    s < 0. ? 0. : s
end

@adjoint relu(s::Real) = relu(s), z -> s < 0. ? (0.,) : (z,)

function conv_net(input::Array{<:Real, 2}, circuit1, circuit2, m)   # first quantum conv layer
    out1 = circuit1 * input
    # convert into 0 and 1s

    # second quantum conv layer
    out2 = circuit2 * out1

    # # maximum pooling
    # v = max_pooling(relu.(out2))

#     # fully connected layer
#     v = m * v

#     # softmax
#     v = softmax(v)
    return dot(m, reshape(out2, length(out2)))
end

# total number of parameters
crs1 = [real_variational_circuit_1d(4, depth) for i in 1:2]
circuit1 = QCNNLayer(crs1, (2, 2), padding=0)
crs2 = [real_variational_circuit_1d(4, depth) for i in 1:1]
circuit2 = QCNNLayer(crs2, (2, 2), padding=0)
m = randn(length(crs1) * length(crs2))

# total number of parameters
x0 = parameters(circuit1, circuit2, m)
# println(x0)
# println(length(parameters(circuit1)))
# println(length(parameters(circuit2)))
# println(length(parameters(m)))
println("total number of parameters $(length(x0))")

loss(c1, c2, ma) = begin
    r = 0.
    for i in 1:length(x_train)
        r += (conv_net(x_train[i], c1, c2, ma) - y_train[i])^2 / 2
    end
    return r
end

@time r = loss(circuit1, circuit2, m)
println("initial loss is $r")

# println("check if the gradient correct? $(check_gradient(loss, circuit1, circuit2, m))")

predict(input::Array{<:Real, 2}) = begin
    r = conv_net(input, circuit1, circuit2, m)
    return r >= 0.5 ? 1 : 0
end

accuracy(x, y) = begin
    r = [predict(x[i])==y[i] for i in 1:length(x)]
    return sum(r)/length(r)
end

# distance(a, b) = dot(a, a) + dot(b, b) - 2*dot(a, b)
# y_pred = [predict(item) for item in x_test]
# score = sum([distance(a, b) for (a, b) in zip(y_pred, y_test)])

println("score before training $(accuracy(x_test, y_test)).")

opt = ADAM(learn_rate)
# opt = Descent(learn_rate)

results = []
los = []
function train(epochs)
    for i in 1:epochs
        @time grad = parameters(gradient(loss, circuit1, circuit2, m))
        Optimise.update!(opt, x0, grad)
        set_parameters!(x0, circuit1, circuit2, m)
        if i % 1 == 0
            ac = accuracy(x_test, y_test)
            ss = loss(circuit1, circuit2, m)
            push!(results, ac)
            push!(results, ss)
            println("accuracy at the $i-th step is $ac, loss is $(ss).")
#             println("loss at step $i is $(loss(circuit1, circuit2, m))")
        end
    end
    return parameters(circuit1, circuit2, m)
end

epochs = 1000

train(epochs)

r = JSON.json(Dict("accuracy"=>results, "loss"=>los))
open("resultd$(depth)rate$(learn_rate).txt", "w") do io
    write(io, r)
end
