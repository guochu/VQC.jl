
# using 2n+4 qubits

const PLUS = [0.5+0.5*im 0.5-0.5*im; 0.5-0.5*im 0.5+0.5*im]
const MINUS = [0.5-0.5*im 0.5+0.5*im; 0.5+0.5*im 0.5-0.5*im]
const CONTROL_PLUS = CONTROL(PLUS)
const CONTROL_MINUS = CONTROL(MINUS)

"""
    n = length(a)
    n qubits
    @ input output
    b[0]
    b[1]
    ...
    b[n-1]
"""
LNNAdderCircuit(a::BinaryInteger) = QCircuit([PHASEGate(i, compute_phase(a, i)) for i in 0:(length(a)-1)])

"""
    n = len(a)
    n+1 qubits
    @input
    c control qubit
    b[0]
    b[1]
    ...
    b[n-1]
    @output
    b[0]
    b[1]
    ...
    b[n-1]
    c control qubit
"""
LNNCAdderCircuit(a::BinaryInteger; inverse::Bool=false) = inverse ? QCircuit(
[gate((i+1, i), SWAP * CONTROL(PHASE(compute_phase(a, i)))) for i in (length(a)-1):-1:0]) : QCircuit(
[gate((i, i+1), SWAP * CONTROL(PHASE(compute_phase(a, i)))) for i in 0:(length(a)-1)])

"""
    this circuit contains n+6 qubits, where n+1 = len(a) = len(N)
    x1
    MS addition (most significant) qubit in |0> out |0>
    x2
    ki qubit in |0> out |0>
    kx qubit
    b[0]
    b[1]
    b[2]
    ...
    b[n]
"""
function LNNModAdderCircuit(N::BinaryInteger, a::BinaryInteger)
    (length(N)==length(a)) || error("binary integer a and N size mismatch.")
	L = length(a)
	n = L-1
	total_circuit = QCircuit()
    push!(total_circuit, gate((3,4), CONTROL_PLUS))
    push!(total_circuit, gate((2,3), CNOT))
    push!(total_circuit, gate((3,4), CONTROL_MINUS))
	push!(total_circuit, gate((2,3), SWAP*CNOT))
	push!(total_circuit, gate((3,4), CONTROL_PLUS))
	append!(total_circuit, shift(LNNCAdderCircuit(a, inverse=false), 4))
	append!(total_circuit, shift(LNNAdderCircuit(N), 4)')
	append!(total_circuit, shift(_QFT(L)', 4) )
	push!(total_circuit, gate((1,2), SWAP))
	push!(total_circuit, gate((2,3), SWAP))
	push!(total_circuit, gate((4,3), SWAP * CNOT))
	append!(total_circuit, move_site(4, n+4))
	append!(total_circuit, shift(_QFT(L), 3))
	append!(total_circuit, shift(LNNCAdderCircuit(N, inverse=true), 3) )
	append!(total_circuit, shift(LNNCAdderCircuit(a, inverse=false)', 4) )
    append!(total_circuit, shift(_QFT(L)', 5) )
	append!(total_circuit, move_site(4, n+5))
	push!(total_circuit, gate(4, X))
	push!(total_circuit, gate((4,3), CNOT))
	push!(total_circuit, gate(4, X))
    push!(total_circuit, gate((2,3), SWAP))
    push!(total_circuit, gate((1,2), SWAP))
	push!(total_circuit, gate((0,1), SWAP))
	append!(total_circuit, shift(_QFT(L), 4) )
	append!(total_circuit, shift(LNNCAdderCircuit(a, inverse=true), 4) )
	push!(total_circuit, gate((3,4), CONTROL_MINUS))
    push!(total_circuit, gate((2,3), CNOT * SWAP))
    push!(total_circuit, gate((3,4), CONTROL_PLUS))
    push!(total_circuit, gate((2,3), CNOT))
	push!(total_circuit, gate((1,2), SWAP))
	push!(total_circuit, gate((0,1), SWAP))
	push!(total_circuit, gate((3,4), CONTROL_MINUS))
	return total_circuit
end

"""
    this circuit contains 2n+4 qubits, where n+1 = len(a) = len(N)
    x[0]
    x[1]
    ...
    x[n-2]
    MS
    x[n-1]
    ki
    kx
    b[0] n+3
    b[1]
    ...
    b[n]
"""
function LNNCModMultiplierCircuit(N::BinaryInteger, a::BinaryInteger)
    (length(N)==length(a)) || error("binary integer a and N size mismatch.")
	L = length(a)

	n = L-1

	total_circuit = QCircuit()

	# qft = shift(_QFT(L), n+3)
    # append!(total_circuit, qft)

	mv = move_site(n-2, 0)

	for i in 0:(n-1)

		apow = modular_multiply(a, 2^i, N)

		madder = LNNModAdderCircuit(N, apow)

        append!(total_circuit, shift(madder, n-2) )

        append!(total_circuit, mv)
    end

    # append!(total_circuit, qft')

	return total_circuit
end

function _cswap()
	circuit = QCircuit()
    push!(circuit, gate((2,1), CNOT))
	push!(circuit, gate((1,2), CONTROL_PLUS))
	push!(circuit, gate((0,1), CNOT))
	push!(circuit, gate((1,2), CONTROL_MINUS))
	push!(circuit, gate((0,1), SWAP * CNOT))
	push!(circuit, gate((1,2), SWAP * CONTROL_PLUS))
	push!(circuit, gate((1,0), CNOT))
	return circuit
end

function mesh(n::Int)
	circuit = QCircuit()
	for i in 0:(n-2)
        append!(circuit, move_site(n+i, 2*i+1))
    end
	return circuit
end

"""
    2n+1 qubits
    c
    a[0]
    a[1]
    ...
    a[n-1]
    b[0]
    b[1]
    ...
    b[n-1]
"""
function control_swap(n::Int)
	total_circuit = QCircuit()
    append!(total_circuit, shift(mesh(n), 1) )
	for i in 0:2:(2*n-1)
        append!(total_circuit, shift(_cswap(), i) )
    end
    append!(total_circuit, mesh(n)')
	return total_circuit
end

"""
    this circuit contains 2n+4 qubits, where n+1 = len(a) = len(N)
    x[0]
    x[1]
    ...
    x[n-2]
    MS
    x[n-1]
    ki
    kx
    b[0]    n+3
    b[1]
    ...
    b[n]
"""
function LNNModUCircuit(N::BinaryInteger, a::BinaryInteger, ra::BinaryInteger)
    ((length(N)==length(a)) && (length(N)==length(ra))) || error("binary integer a, ar and N size mismatch.")
	L = length(a)
	n = L-1

	total_circuit = QCircuit()

    append!(total_circuit, LNNCModMultiplierCircuit(N, a))

    append!(total_circuit, shift(_QFT(L)', n+3) )
	append!(total_circuit, move_site(n-1, 0))
	append!(total_circuit, move_site(n+2, 1))
	append!(total_circuit, move_site(n+2, 2))
	append!(total_circuit, shift(move_site(0, n), n+3))

	# controled swap
    append!(total_circuit, shift(control_swap(n), 2) )

    append!(total_circuit, move_site(1, n+1))
	append!(total_circuit, move_site(0, n-1))
	append!(total_circuit, move_site(2*n+2, n+1))
	append!(total_circuit, move_site(2*n+3, n+3))
	append!(total_circuit, shift(_QFT(L), n+3))

    append!(total_circuit, LNNCModMultiplierCircuit(N, ra)')

	return total_circuit
end


"""
    compute fa,N (x) = a^x mod N
    2n+3+m qubits alined as follows
    f[0]
    f[1]
    ...
    f[m-1]
    x[0]   m
    x[1]
    ...
    x[n-2]
    MS
    x[n-1]  m+n
    --ki--
    kx      m+n+1
    b[0]   m+n+2
    b[1]
    ...
    b[n]
    @input
    @x m bits qnumber
    @b result: n bits qnumber, initialized to be |0>
    @output
    @b->(a^x)%N
"""
function LNNModExponentiatorCircuit(N::BinaryInteger, a::BinaryInteger, m::Int)
	(length(N)==length(a)) || error("binary integer a and N size mismatch.")

	total_circuit = QCircuit()

	L = length(N)

	n = L-1

	ra = modular_inverse(a, N)

	qft_circuit = shift(_QFT(L), m+n+2)

    append!(total_circuit, qft_circuit)

	for i in (m-1):-1:0

		mv = move_site(i, m+n)

        append!(total_circuit, mv)

		# central block operations
        append!(total_circuit, shift(LNNModUCircuit(N, a, ra), m-1))

        append!(total_circuit, mv')

		a = modular_multiply(a, a, N)

        ra = modular_multiply(ra, ra, N)
    end

    append!(total_circuit, qft_circuit')

	return total_circuit
end
