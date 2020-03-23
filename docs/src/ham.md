# Hamiltonian


VQC also has preliminary suport for hamiltonian evolution.

## A simple hamiltonian simulation
```@example
push!(LOAD_PATH, "../../src")
using VQC
ps = spin_half()
pb = boson(d=4)
ham = Hamiltonian([ps, pb])

add!(ham, (1,2), ("sp", "a"), coeff=1)
add!(ham, (1,2), ("sm", "adag"), coeff=1)
add!(ham, (1,), ("sz",), coeff=0.5)
add!(ham, (2,), ("n",), coeff=2)

state = kron(spin_half_state(0), fock_state(4, 2))

# unitary evolution
state = apply(ham, 0.5, state)

# Create an observer
observer = Observers(ham)
add!(observer, (1,), ("sz",), name="z")
add!(observer, (2,), ("n",), name="n")
add!(observer, (1,2), ("sp", "a"), name="j")
obs = apply(observer, state)
```