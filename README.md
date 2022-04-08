---

## Introduction:
  VQC is an open source framework that can simulate variational quantum circuits and used for quantum machine learning tasks.
  * **Simple but powerful**. VQC supports any single-qubit, two-qubit, three-qubit gate operations, as well as measurements. The same quantum circuit can be used as variational quantum circuits almost for free.

  * **Everything is differentiable**. Not only the quantum circuit, the quantum state itself is also differentiable, almost without any changing of code. In most of the cases, user can write a very complex expression built on top of the quantum circuit and the quantum state, and the whole expression will be differentiable.

  * **Flexiable operations on quantum gates and quantum circuits**. Quantum circuit and quantum gates suport operations such as adjoint, transpose, conjugate, shift to make life easier when building very complex circuits.

  * **Zygote as backend for auto differentiation**. VQC use Zygote as backend for auto differentiation.


## Todo list;
* Change the documentations
* Compute expectation value on density matrix
* AD for density matrix related operations
* Support for general quantum channel operation on density matrix



