module VQC

using Zygote
using Zygote: @adjoint

import Base.+, Base.-, Base.*, Base./
import LinearAlgebra: dot, norm, ishermitian

using KrylovKit: exponentiate
using SparseArrays: spzeros, sparse, SparseMatrixCSC
using Logging: @warn


include("constants.jl")
include("misc/misc.jl")

include("defs.jl")
include("state.jl")
include("gate/gate.jl")

include("circuit/circuit.jl")
include("observer/observer.jl")

# differentiation
include("diff/differentiation.jl")
include("diff/autodiff.jl")

include("ham/ham.jl")
include("ctrlham/ctrlham.jl")

# utility functions
include("utility/utility.jl")

# chain operation
include("chain.jl")

end