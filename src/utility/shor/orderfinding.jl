

"""
    n = len(a)-1 = len(N)-1
    4*n+2 qubits is needed
"""
function order_finding_qft_circuit(a::BinaryInteger, N::BinaryInteger; verbosity::Int=1)

    ai = get_value(a)
	Ni = get_value(N)
	(ai==1) && error("the input a can not be 1.")
	(Ni==1) && error("the input N can not be 1.")

	(gcd(ai, Ni)==1) || error("a and N are not co-prime to each other.")
    # nmax = length(a)
	# precision = 2**(-2*nmax-1)
	n = length(a)
	L = n+1
	a = BinaryInteger(get_value(a), L)
	N = BinaryInteger(get_value(N), L)
    (verbosity >= 3) && println("n=$n, total number of qubits $(4*n+2).")


	circuit = QCircuit()
    push!(circuit, gate(2*n+1, X))
    for i in (2*n+2):(4*n+1)
        push!(circuit, gate(i, H))
    end
	# circuit = shift(circuit, 2*n+2)
    append!(circuit, QFTCModExponentiatorCircuit(N, a, 2*n))
    append!(circuit, shift(_QFT(2*n), 2*n+2)')

    return circuit
	# check N
end

function order_finding_by_qft(::Type{T}, a::Int, N::Int; verbosity::Int=1) where {T<:Number}
    L = max(length(digits(a, base=2)), length(digits(N, base=2)))
    state = StateVector(T, 4*L+2)
    a = BinaryInteger(a, L)
    N = BinaryInteger(N, L)

    circuit = shift(order_finding_qft_circuit(a, N, verbosity=verbosity), 1)
    # println("max min position $(max_site(circuit.data)), $(min_site(circuit.data)).")

    # circuit = fuse_gate(circuit)
    apply_circuit!(circuit, state)
    r = Int[]
    for i in 0:(2*L-1)
        observer = QMeasure(2*L+i+3)
        istate, probability = apply_circuit!(observer, state)
        push!(r, istate)
    end
    # return sum([(r[2 * L - i]*1. / (1 << (i + 1))) for i in 0:(2 * L-1)])
    return to_digits(r)
end

"""
    see reference
    "Implementation of Shor’s Algorithm on a Linear Nearest Neighbour Qubit Array"
    n = len(a)-1 = len(N)-1
    4*n+3 qubits is needed
    f[0]
    f[1]
    ...
    f[2n-1]
    x[0] 2n
    x[1]
    ...
    x[n-2]
    MS
    x[n-1]  3n
    kx
    b[0]
    b[1]
    ...
    b[n]
"""
function order_finding_lnn_circuit(a::BinaryInteger, N::BinaryInteger; verbosity::Int=1)
    ai = get_value(a)
	Ni = get_value(N)
	(ai==1) && error("the input a can not be 1.")
	(Ni==1) && error("the input N can not be 1.")

	(gcd(ai, Ni)==1) || error("a and N are not co-prime to each other.")
    # nmax = length(a)
	# precision = 2**(-2*nmax-1)
	n = length(a)
	L = n+1
	a = BinaryInteger(get_value(a), L)
	N = BinaryInteger(get_value(N), L)
    (verbosity >= 3) && println("n=$n, total number of qubits $(4*n+3).")

	circuit = QCircuit()
    push!(circuit, gate(3*n, X))
	for i in 0:(2*n-1)
        push!(circuit, gate(i, H))
    end
    append!(circuit, LNNModExponentiatorCircuit(N, a, 2*n))
	append!(circuit, _QFT(2*n)')
    append!(circuit, move_site(3*n, 3*n-1))

    return circuit
end

function order_finding_by_lnn(::Type{T}, a::Int, N::Int; verbosity::Int=1) where {T<:Number}
    L = max(length(digits(a, base=2)), length(digits(N, base=2)))
    state = StateVector(T, 4*L+3)
    a = BinaryInteger(a, L)
    N = BinaryInteger(N, L)
    circuit = shift(order_finding_lnn_circuit(a, N, verbosity=verbosity), 1)

    # println("max min position $(max_site(circuit.data)), $(min_site(circuit.data)).")

    # circuit = fuse_gate(circuit)
    apply_circuit!(circuit, state)

    # # testing
    # println("testing....")
    # for i in (3*L+1):(4*L+3)
    #     observer = QMeasure(i)
    #     istate, probability = apply!(observer, state)
    #     println("get $istate with probability $probability at $i-th qubit.")
    # end

    r = Int[]
    for i in 0:(2*L-1)
        observer = QMeasure(i+1)
        istate, probability = apply_circuit!(observer, state)
        push!(r, istate)
    end
    # return sum([(r[2 * L - i]*1. / (1 << (i + 1))) for i in 0:(2 * L-1)])
    return to_digits(r)
end

function order_finding(::Type{T}, a::Int, N::Int; less_qubits::Bool=true, algorithm::Symbol=:lnn, verbosity::Int=1) where {T<:Number}
    if less_qubits
        return order_finding_scq(T, a, N; algorithm=algorithm, verbosity=verbosity)
    else
        if algorithm == :lnn
            return order_finding_by_lnn(T, a, N; verbosity=verbosity)
        elseif algorithm == :qft
            return order_finding_by_qft(T, a, N; verbosity=verbosity)
        else
            error("algorithm $algorithm not implemented.")
        end
    end
end

order_finding(a::Int, N::Int) = order_finding(ComplexF64, a, N)
