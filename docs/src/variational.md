# Variational Quantum Circuit

In this section we will provide a simple pipeline to demonstrate how to
use VQC to build variational quantum circuits


## Creating a variational quantum circuit
```@repl
push!(LOAD_PATH, "../../src")
using VQC
L = 3
depth = 2
circuit = QCircuit()
for i in 1:L
	push!(circuit, RzGate(i, Variable(rand())))
	push!(circuit, RyGate(i, Variable(rand())))
	push!(circuit, RzGate(i, Variable(rand())))
end
for l in 1:depth
	for i in 1:L-1
		push!(circuit, CNOTGate((i, i+1)))
	end
	for i in 1:L
		push!(circuit, RzGate(i, Variable(rand())))
		push!(circuit, RyGate(i, Variable(rand())))
		push!(circuit, RzGate(i, Variable(rand())))
	end
end	
# all the parameters of the circuit
paras = parameters(circuit)

# reset all the parameters of the circuit
new_paras = zeros(length(paras))
set_parameters!(new_paras, circuit)
paras = parameters(circuit)

# compute the gradient
using Zygote
target_state = qrandn(L)
initial_state = qstate(L)
loss(c) = distance(target_state, c * initial_state)
grad = gradient(loss, circuit)
```
The above definition has been predefined by the function variational_circuit
```@docs
variational_circuit(L::Int, depth::Int, g::Function=rand)
```

## A simple application of VQC and Flux
```@repl
using VQC
using Zygote
using Flux.Optimise

circuit = variational_circuit(3, 2)
target_state = (sqrt(2)/2) *  (qstate([1,1,1]) + qstate([0,0,0]))
initial_state = qstate([0,0,0])
loss(c) = distance(target_state, c * initial_state)

opt = ADAM()
epoches = 10
x0 = parameters(circuit)
for i in 1:epoches
	grad = collect_variables(gradient(loss, circuit))
	Optimise.update!(opt, x0, grad)
	set_parameters!(x0, circuit)
	println("loss value at $i-th iteration is $(loss(circuit)).")
end
```

## Some utility functions
```@docs
collect_variables(args...)
parameters(args...)
set_parameters!(coeff::AbstractVector{<:Number}, args...)
simple_gradient(f, args...; dt::Real=1.0e-6)
check_gradient(f, args...; dt::Real=1.0e-6, atol::Real=1.0e-4, verbose::Int=0)
```

