
push!(LOAD_PATH, "../../src")

include("qcnn.jl")

using MLDatasets
using Statistics: mean
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

function squeeze(x::AbstractMatrix)
    r = zeros(7, 7)
    for j in 1:7
        for i in 1:7
            r[i, j] = mean(x[4*(i-1)+1:4*i, 4*(j-1)+1:4*j])
        end
    end
    return r
end

function select_indexes(x, y, n::Int)
    x_less = []
    y_less = []
    for (i, item) in enumerate(y)
        if item in 0:n-1
            push!(x_less, squeeze(x[i]))
            push!(y_less, item)
        end
    end
    return [x_less...], [y_less...]
end

function shuffle_data(x, y)
    n = collect(1:length(x))
    shuffle!(n)
    return x[n], y[n]
end

function read_mnist()
    x_train, y_train = MNIST.traindata()
	x_test, y_test  = MNIST.testdata()

	x_train = [x_train[:, :, i]/norm(x_train[:, :, i]) for i in 1:size(x_train, 3)]
	x_test = [x_test[:, :, i]/norm(x_test[:, :, i]) for i in 1:size(x_test, 3)]

    return x_train, y_train, x_test, y_test
end

function processing_data(x_train, y_train, x_test, y_test, n::Int)
    x_train, y_train = select_indexes(x_train, y_train, n)
    x_test, y_test = select_indexes(x_test, y_test, n)
    y_train = [Flux.onehot(item, 0:n-1) for item in y_train]
    y_test = [Flux.onehot(item, 0:n-1) for item in y_test]
    x_train, y_train = shuffle_data(x_train, y_train)
    m = 250 * n
    x_train, y_train = x_train[1:m], y_train[1:m]
    x_test, y_test = shuffle_data(x_test, y_test)
    x_test, y_test = x_test[1:m], y_test[1:m]
    println("number of training $(length(x_train)), number of testing $(length(x_test)).")
    println("start training...")
    return x_train, y_train, x_test, y_test
end

function prepare_mnist_data(nlabel::Int)
    x_train, y_train, x_test, y_test = read_mnist()
    return processing_data(x_train, y_train, x_test, y_test, nlabel)
end


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

	# # maximum pooling
    out2 = max_pooling(out1, (3,3), 0)

    # second quantum conv layer
    out3 = circuit2 * out2
    # println(size(out2))


#     # softmax
#     v = softmax(v)
    return softmax(m * reshape(out3, length(out3)))
end

distance(x::AbstractVector, y::AbstractVector) = dot(x, x) + dot(y, y) - 2 * dot(x, y)

function train_single(nlabel::Int, nitr::Int, id::Int, learn_rate::Real=0.01, depth::Int=9)
    println("parameters nlabel=$nlabel, epochs=$nitr, learn rate=$learn_rate, circuit depth=$depth, index=$id.")

    x_train, y_train, x_test, y_test = prepare_mnist_data(nlabel)

    crs1 = [real_variational_circuit_1d(3*3, depth) for i in 1:2]
    circuit1 = QCNNLayer(crs1, (3, 3), padding=0)
    crs2 = [real_variational_circuit_1d(3*3, depth) for i in 1:2]
    circuit2 = QCNNLayer(crs2, (3, 3), padding=0)
    m = randn(2, length(crs1) * length(crs2))
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

    function train(epochs, index::Int)
        results = []
        los = []
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
        filename = "result/MNISTnlabel$(nlabel)alpha$(learn_rate)epochs$(epochs)index$(index).json"
        println("save data to path $path_name.")
        r = JSON.json(Dict("accuracy"=>results, "loss"=>los))
        open(filename, "w") do f
            write(f, r)
        end
    end

    train(nitr, id)

end

function main(paras)
    nlabel = parse(Int, get(paras, "nlabel", "2"))
    index = parse(Int, get(paras, "index", "1"))
    epochs = parse(Int, get(paras, "epoch", "500"))
    alpha = parse(Float64, get(paras, "alpha", "0.01"))
    return train_single(nlabel, epochs, index, alpha)
end

function parse_cmd_line_args(args::Vector{<:AbstractString}, s::AbstractString=":")
	r = Dict{String, String}()
	for arg in args
		k, v = split(arg, s)
		r[k] = v
	end
	return r
end

paras = parse_cmd_line_args(ARGS)
main(paras)
