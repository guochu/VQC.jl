# sqc : single qubit control to use less qubits

function to_digits(s::Vector)
	r = 0.
	for i in 0:(length(s)-1)
		r = r + s[i+1]*2.0^(-i-1)
    end
	return r
end


apply_circuit!(gt, state) = apply!(gt, state)

"""
    see reference
    using the one control qubit trick
    this circuit contains 2n+3 qubits, where n+1 = len(a) = len(N)
    t qubit in |0> out |0>
    b[0] addition (most significant) qubit in |0> out |0>
    b[1]
    b[2]
    ...
    b[n]
    x[0]
    x[1]
    ...
    x[n-1]
    c control qubit
"""
function order_finding_by_qft_impl(a::BinaryInteger, N::BinaryInteger, state; verbosity::Int=1)
    (length(N)==length(a)) || error("binary integer a and N size mismatch.")

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
    (verbosity >= 3) && println("n=$n, total number of qubits $(2*n+3).")
	control_site = 2*n+3
	observer = QMeasure(control_site, auto_reset=true)

	ra = modular_inverse(a, N)

	qft_circuit = shift(_QFT(L), 2)

	apply_circuit!(qft_circuit, state)

	Hgate = gate(control_site, H)

	apply_circuit!(Hgate, state)

	astore = []
	rastore = []
	for i in 1:(2*n)
        push!(astore, a)
        push!(rastore, ra)
		a = modular_multiply(a, a, N)
		ra = modular_multiply(ra, ra, N)
    end

	r = []
	# print('total number of measurements %s'%(2*n))
	for i in 0:(2*n-1)
		apply_circuit!(Hgate, state)

		circuit = shift(QFTModUCircuit(N, astore[2*n-i], rastore[2*n-i]), 1)

        # println("max min site $(max_site(circuit.data)), $(min_site(circuit.data)).")

		# println("number of gates $(length(circuit)).")
		# circuit = fuse_gate(circuit)
		# println("number of gates after gate fusion $(length(circuit)).")

		apply_circuit!(circuit, state)

		# apply rorate
		rot = gate(control_site, [1. 0.; 0. exp(-pi*im*to_digits(reverse(r)))])
		apply_circuit!(rot, state)

		apply_circuit!(Hgate, state)

		istate, probability = apply_circuit!(observer, state)

        push!(r, istate)

		(verbosity >= 2) && println("measurement outcome at $i-th step is $istate.")
    end

	y = sum([(r[2 * n - i]*1. / (1 << (i + 1))) for i in 0:(2 * n-1)])
	return y

	# numerator, d = continuous_fraction(y, Ni-1)
	# (verbosity >= 1) && println("numerator is $numerator, denominator is $d.")
    #
	# (modular_pow(ai, d, Ni)==1) && return d
    # (verbosity >= 2) && println("order finding failed.")
	#
	# println("fraction is $y.")
	# return -1
end

function scq_order_finding_by_qft(::Type{T}, a::Int, N::Int; verbosity::Int=1) where {T<:Number}
    L = max(length(digits(a, base=2)), length(digits(N, base=2)))
    state = StateVector(T, 2*L+3)
    apply_circuit!(gate(2*L+2, X), state)
    return order_finding_by_qft_impl(BinaryInteger(a, L), BinaryInteger(N, L), state; verbosity=verbosity)
end

"""
	see reference
	using the one control qubit trick
	"Implementation of Shor’s Algorithm on a Linear Nearest Neighbour Qubit Array"
	n = len(a)-1 = len(N)-1
	2*n+4 qubits is needed
	x[0]
	x[1]
	...
	x[n-2]
	MS
	x[n-1]  n
	ki
	kx
	b[0]
	b[1]
	...
	b[n]
"""
function order_finding_by_lnn_impl(a::BinaryInteger, N::BinaryInteger, state; verbosity::Int=1)
	# assert(driver in ['lnn'])
	ai = get_value(a)
	Ni = get_value(N)
	(ai==1) && error("the input a can not be 1.")
	(Ni==1) && error("the input N can not be 1.")

	(gcd(ai, Ni)==1) || error("a and N are not co-prime to each other.")
	# precision = 2**(-2*nmax-1)
	n = length(a)
	L = n+1

	a = BinaryInteger(get_value(a), L)
	N = BinaryInteger(get_value(N), L)
	(verbosity >= 3) && println("n=$n, total number of qubits $(2*n+4).")

	observer = QMeasure(n+2, auto_reset=true)

	ra = modular_inverse(a, N)

	qft_circuit = shift(_QFT(L), n+4)

	apply_circuit!(qft_circuit, state)

	Hgate = gate(n+2, H)

	apply_circuit!(Hgate, state)

	astore = []
	rastore = []
	for i in 1:2*n
		push!(astore, a)
        push!(rastore, ra)
		a = modular_multiply(a, a, N)
		ra = modular_multiply(ra, ra, N)
	end

	r = []
	for i in 0:(2*n-1)
		apply_circuit!(Hgate, state)

		circuit = shift(LNNModUCircuit(N, astore[2*n-i], rastore[2*n-i]), 1)

		# println("max min site $(max_site(circuit.data)), $(min_site(circuit.data)).")

		# println("number of gates $(length(circuit)).")
		# circuit = fuse_gate(circuit)
		# println("number of gates after gate fusion $(length(circuit)).")

		apply_circuit!(circuit, state)

		# apply rorate
		rot = gate(n+2, [1. 0.; 0. exp(-pi*im*to_digits(reverse(r)))])
		apply_circuit!(rot, state)

		apply_circuit!(Hgate, state)

		istate, probability = apply_circuit!(observer, state)

		push!(r, istate)

		(verbosity >= 2) && println("measurement outcome at $i-th step is $istate.")
	end

	y = sum([(r[2 * n - i]*1. / (1 << (i + 1))) for i in 0:(2 * n-1)])
	return y
	# println("fraction is $y.")
	# return -1
end

function scq_order_finding_by_lnn(::Type{T}, a::Int, N::Int; verbosity::Int=1) where {T<:Number}
    L = max(length(digits(a, base=2)), length(digits(N, base=2)))
    state = StateVector(T, 2*L+4)
    apply_circuit!(gate(L+1, X), state)
    return order_finding_by_lnn_impl(BinaryInteger(a, L), BinaryInteger(N, L), state; verbosity=verbosity)
end

"""
	quantum order finding algorithm using single control qubits technique.
"""
function order_finding_scq(::Type{T}, a::Int, N::Int; algorithm::Symbol=:lnn, verbosity::Int=1) where {T<:Number}
	if algorithm == :lnn
		return scq_order_finding_by_lnn(T, a, N; verbosity=verbosity)
	elseif algorithm == :qft
		return scq_order_finding_by_qft(T, a, N; verbosity=verbosity)
	else
		error("algorithm $algorithm not implemented.")
	end
end
