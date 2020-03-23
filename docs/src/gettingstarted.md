# Getting Started

In this section we will provide a simple pipeline to demonstrate how to
use VQC to build quantum computing applications


Pipeline for quantum circuit simulation

## Initialize a quantum state
Definition of function qstate
```@docs
qstate(::Type{T}, thetas::AbstractVector{<:Real}) where {T <: Number}
qstate(thetas::AbstractVector{<:Real})
qstate(::Type{T}, n::Int) where {T <: Number}
qstate(n::Int) 
```
Extract amplitudes from quantum state
```@docs
amplitude(s::AbstractVector, i::AbstractVector{Int}) 
amplitudes(s::AbstractVector)
```


Examples
```@example
push!(LOAD_PATH, "../../src")
using VQC
state = qstate(2)
state = qstate([1, 0])
state = qstate([0.5, 0.7])
```


## Quantum gate
Predefined elementary gates "X, Y, Z, S, H, sqrtX, sqrtY, T, Rx, Ry, Rz, CONTROL, CZ, CNOT, CX, SWAP, iSWAP, XGate, YGate, ZGate, HGate, SGate, TGate, SqrtXGate, SqrtYGate, RxGate, RyGate, RzGate, CZGate, CNOTGate, SWAPGate, iSWAPGate, CRxGate, CRyGate, CRzGate, TOFFOLIGate"

```@example
using VQC

circuit = QCircuit()

# standard one-qubit gate
push!(circuit, (1, H))
empty!(circuit)
push!(circuit, HGate(1))
empty!(circuit)
push!(circuit, gate(1, H))
empty!(circuit)
# standard two-qubit gate
push!(circuit, ((1, 2), CZ))
empty!(circuit)
push!(circuit, CZGate((1, 2)))
empty!(circuit)
push!(circuit, gate((1,2), CZ))
empty!(circuit)
# a parameteric one-qubit gate
push!(circuit, RxGate(1, Variable(0.5)))
empty!(circuit)
# a parameteric two-qubit gate
push!(circuit, CRxGate((1,2), Variable(0.5)))
empty!(circuit)
# This will create a non-parameteric gate instead
push!(circuit, RxGate(1, 0.5))
```

## Quantum circuit
Adding new gates
```@docs
add!(x::AbstractCircuit, s)
Base.push!(x::AbstractCircuit, s::AbstractGate)
Base.append!(x::AbstractCircuit, y::AbstractCircuit)
Base.append!(x::AbstractCircuit, y::Vector{T}) where {T<:AbstractGate}
```

Circuit manipulations
```@example
using VQC
circuit = QCircuit()
push!(circuit, (1, H))
push!(circuit, ((1, 2), CZ))
c1 = transpose(circuit)
c2 = conj(circuit)
c3 = circuit'
```

## Apply quantum circuit to state
```@docs
apply!(circuit::AbstractCircuit, v::Vector)
*(circuit::AbstractCircuit, v::AbstractVector)
*(v::AbstractVector, circuit::AbstractCircuit)
```

## Quantum measurement
Measure and collapse a quantum state
```@docs
measure(qstate::AbstractVector, pos::Int)
measure!(qstate::AbstractVector, pos::Int; auto_reset::Bool=true)
```

Postselection
```@docs
post_select!(qstate::AbstractVector, key::Int, state::Int=0)
post_select(qstate::AbstractVector, key::Int, state::Int=0; keep::Bool=false)
```




