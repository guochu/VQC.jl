# Quantum Control



VQC support variational hamiltonian evolutions.

```@docs
ctrlham(a::AbstractMatrix, b::Vector{<:AbstractMatrix}, nparas::Int)
```

The variational hamiltonian object can be used in the same way as a variational quantum circuit
```@example
push!(LOAD_PATH, "../../src")
using VQC
using Zygote

random_hermitian(n) = begin
	a = randn(n, n)
	return a + a'
end

H0 = random_hermitian(2)
H1 = [random_hermitian(2) for i in 1:2] 
h = ctrlham(H0, H1, 5)

initial_state = [1, 0]
target_state = randn(2)

loss(c) = distance(target_state, c * initial_state)

grad = gradient(loss, h)

# Check the gradient
check_gradient(loss, h)
```