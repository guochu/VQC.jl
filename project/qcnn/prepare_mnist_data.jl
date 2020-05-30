

using MLDatasets
using Statistics: mean
using Random: shuffle!
using LinearAlgebra: norm
using JSON

function squeeze(x::AbstractMatrix)
    r = zeros(7, 7)
    for j in 1:7
        for i in 1:7
            r[i, j] = mean(x[4*(i-1)+1:4*i, 4*(j-1)+1:4*j])
        end
    end
    return r
end

function shuffle_data(x, y)
    n = collect(1:length(x))
    shuffle!(n)
    return x[n], y[n]
end

function select_indexes(x, y, n::Int)
    x_less = []
    y_less = []
    count = zeros(Int, 10)
    for (i, item) in enumerate(y)
        if count[item+1] < n
            push!(x_less, squeeze(x[i]))
            push!(y_less, item)
            count[item+1] += 1
        end
    end
    return [x_less...], [y_less...]
end

function read_mnist()
    x_train, y_train = MNIST.traindata()
	x_test, y_test  = MNIST.testdata()

	x_train = [x_train[:, :, i]/norm(x_train[:, :, i]) for i in 1:size(x_train, 3)]
	x_test = [x_test[:, :, i]/norm(x_test[:, :, i]) for i in 1:size(x_test, 3)]

    return x_train, y_train, x_test, y_test
end

function processing_data(x_train, y_train, x_test, y_test, n::Int=250)
    x_train, y_train = shuffle_data(x_train, y_train)
    x_test, y_test = shuffle_data(x_test, y_test)

    x_train, y_train = select_indexes(x_train, y_train, n)
    x_test, y_test = select_indexes(x_test, y_test, n)

    # println(y_train)
    # println(y_test)

    result = JSON.json(Dict("xtrain"=>x_train, "ytrain"=>y_train, "xtest"=>x_test, "ytest"=>y_test))
    open("mnist_data.txt", "w") do f
        write(f, result)
    end
end

function prepare_mnist_data()
    x_train, y_train, x_test, y_test = read_mnist()
    processing_data(x_train, y_train, x_test, y_test, 250)
end

prepare_mnist_data()
