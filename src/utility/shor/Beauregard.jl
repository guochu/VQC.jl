
# using 2n+3 qubits

# zero based circuit
# QFTAdderCircuit(a::BinaryInteger) = QCircuit([gate(i, rotationA(a, i)) for i in 0:(length(a)-1)])
QFTAdderCircuit(a::BinaryInteger) = QCircuit([PHASEGate(i, compute_phase(a, i)) for i in 0:(length(a)-1)])

"""
    n = length(a)
    n+1 qubits
    b[0]
    b[1]
    ...
    b[n-1]
    c control qubit
"""
# QFTCAdderCircuit(c::Int, a::BinaryInteger) = QCircuit([gate((c, i), CONTROL(rotationA(a, i))) for i in 0:(length(a)-1)])
# QFTCAdderCircuit(c::Int, a::BinaryInteger) = QCircuit(
# [CONTROLGate((c, i), rotationA(a, i)) for i in 0:(length(a)-1)])
QFTCAdderCircuit(c::Int, a::BinaryInteger) = QCircuit(
[CPHASEGate((c, i), compute_phase(a, i)) for i in 0:(length(a)-1)])

"""
    n = length(a)
    n+2 qubits
    b[0]
    b[1]
    ...
    b[n-1]
    c2 control qubit 2
    c1 control qubit 1
"""
# QFTCCAdderCircuit(c1::Int, c2::Int, a::BinaryInteger) = QCircuit(
# [gate((c1, c2, i), CONTROLCONTROL(rotationA(a, i))) for i in 0:(length(a)-1)])
QFTCCAdderCircuit(c1::Int, c2::Int, a::BinaryInteger) = QCircuit(
[CCPHASEGate((c1, c2, i), compute_phase(a, i)) for i in 0:(length(a)-1)])

"""
    this circuit contains n+4 qubits, where n+1 = length(a) = length(N)
    t qubit in |0> out |0>
    b[0] addition (most significant) qubit in |0> out |0>
    b[1]
    b[2]
    ...
    b[n]
    c2
    c1
"""
function QFTCCModAdderCircuit(N::BinaryInteger, a::BinaryInteger)
    (length(N)==length(a)) || error("binary integer a and N size mismatch.")
	L = length(a)
	n = L-1

	total_circuit = QCircuit()

	cc_adder = shift(QFTCCAdderCircuit(L+1, L, a), 1)

	c_adder = shift(QFTCAdderCircuit(-1, N), 1)

	adder = shift(QFTAdderCircuit(N), 1)

	qft = shift(_QFT(L), 1)

    append!(total_circuit, cc_adder)

    append!(total_circuit, adder')

    append!(total_circuit, qft')

    # push!(total_circuit, gate((1, 0), CNOT))
    push!(total_circuit, CNOTGate(1, 0))

    append!(total_circuit, qft)

    append!(total_circuit, c_adder)

    append!(total_circuit, cc_adder')

    append!(total_circuit, qft')

    # push!(total_circuit, gate(1, X))
    push!(total_circuit, XGate(1))

    # push!(total_circuit, gate((1, 0), CNOT))
    push!(total_circuit, CNOTGate(1, 0))

    # push!(total_circuit, gate(1, X))
    push!(total_circuit, XGate(1))

    append!(total_circuit, qft)

    append!(total_circuit, cc_adder)

	return total_circuit
end

"""
    this circuit contains 2n+3 qubits, where n+1 = length(a) = length(N)
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
function QFTCModMultiplierCircuit(N::BinaryInteger, a::BinaryInteger)

	(length(N)==length(a)) || error("binary integer a and N size mismatch.")
	L = length(a)

	n = L-1

	total_circuit = QCircuit()

	mv1 = move_site(2*n+2, n+3)

	for i in 0:(n-1)
		mv2 = shift(move_site(n-1-i, 0), n+2)

		append!(total_circuit, mv2)

		append!(total_circuit, mv1)

		apow = modular_multiply(a, 2^i, N)

		madder = QFTCCModAdderCircuit(N, apow)

		append!(total_circuit, madder)

		append!(total_circuit, mv1')

		append!(total_circuit, mv2')
    end

	return total_circuit
end

"""
    this circuit contains 2n+3 qubits, where n+1 = length(a) = length(N)
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
function QFTModUCircuit(N::BinaryInteger, a::BinaryInteger, ra::BinaryInteger)

	L = length(a)
	n = L-1

	qft = shift(_QFT(L), 1)

	c_swap = control_swap_block(2*n+2, n, 2, 1, n+2, 1)

	total_circuit = QCircuit()

	mult_circuit = QFTCModMultiplierCircuit(N, a)

	append!(total_circuit, mult_circuit)

	append!(total_circuit, qft')

	append!(total_circuit, c_swap)

	append!(total_circuit, qft)

	rmult_circuit = QFTCModMultiplierCircuit(N, ra)

	append!(total_circuit, rmult_circuit')

	return total_circuit
end

"""
    compute fa,N (x) = a^x mod N
    2n+2+m qubits alined as follows
    t qubit in |0> out |0>
    b[0] addition qubit int |0> out |0>
    b[1]
    ...
    b[n]
    t[0]
    t[1]
    ...
    t[n-1]
    x[0]
    x[1]
    ...
    x[m-1]
    @input
    @t n bits temporary qnumber, initialized to be |1>
    @x m bits qnumber
    @b result: n bits qnumber, initialized to be |0>
    @output
    @b->(a^x)%N
"""
function QFTCModExponentiatorCircuit(N::BinaryInteger, a::BinaryInteger, m::Int)
	(length(N)==length(a)) || error("binary integer a and N size mismatch.")

	L = length(N)

	n = L-1

    ra = modular_inverse(a, N)

	total_circuit = QCircuit()

	c_swap = control_swap_block(2*n+2, n, 2, 1, n+2, 1)

	qft = shift(_QFT(L), 1)

    append!(total_circuit, qft)


	for i in (m-1):-1:0

		mv = shift(move_site(i, 0), 2*n+2)

        append!(total_circuit, mv)

		# central block operations

		mult_circuit = QFTCModMultiplierCircuit(N, a)

        append!(total_circuit, mult_circuit)

        append!(total_circuit, qft')

        append!(total_circuit, c_swap)

        append!(total_circuit, qft)

		rmult_circuit = QFTCModMultiplierCircuit(N, ra)

        append!(total_circuit, rmult_circuit')

        append!(total_circuit, mv')

        a = modular_multiply(a, a, N)

        ra = modular_multiply(ra, ra, N)
    end

    append!(total_circuit, qft')

	return total_circuit
end
