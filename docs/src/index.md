

Variational quantum circuit simulator in julia

# VQC.jl's documentation

VQC is a julia library for variational quantum circuit simulations. VQC 
aims to supprot automatic differentiation and allow users to easily build 
hybrid quantum-classical algorithms. Current VQC also has a basic support 
for variational Hamiltonian simulation.

A simple code snippet to create a two-qubit bell state
```@example
push!(LOAD_PATH, "../../src")
using VQC

state = qstate(2)
circuit = QCircuit()
push!(circuit, (1, H))
push!(circuit, ((1, 2), CNOT))

apply!(circuit, state)
amplitudes(state)

# Perform quantum measurement
i, prob = measure!(state, 1)
println("probability of the 1-th qubit in state $i is $prob.")

# Obtain a particular amplitude
p = amplitude(state, [0, 1])
```

Build a variational quantum circuit is as simple as a normal quantum state
```@example
using VQC
using Zygote
L = 3
state = qstate(L)
circuit = QCircuit()
for i in 1:L
	push!(circuit, RzGate(i, Variable(rand())))
	push!(circuit, RyGate(i, Variable(rand())))
	push!(circuit, RzGate(i, Variable(rand())))
end

for depth in 1:2
	for i in 1:L-1
		push!(circuit, CNOTGate((i, i+1)))
	end
	for i in 1:L
		push!(circuit, RzGate(i, Variable(rand())))
		push!(circuit, RxGate(i, Variable(rand())))
		push!(circuit, RzGate(i, Variable(rand())))
	end
end

# Create a random quantum state of 3 qubits
target_state = qrandn(L)
loss(c) = distance(target_state, c * state)
v = loss(circuit)
grad = gradient(loss, circuit)
check_gradient(loss, circuit)
```


Start to use Meteor.jl from "Getting Started" section.

```@contents
Pages = ["gettingstarted.md"]
Depth = 2
```

```@contents
Pages = ["variational.md"]
Depth = 2
```

```@contents
Pages = ["ham.md"]
Depth = 2
```

```@contents
Pages = ["qctrl.md"]
Depth = 2
```
