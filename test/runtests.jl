using Test
using LinearAlgebra
using VQC
using QuantumCircuits
using QuantumCircuits.Hamiltonian: PauliTerm, PauliSum

# 测试辅助：把局域矩阵 M（qs[1] = 矩阵最高位，1-based 比特号）嵌入 n 比特空间（小端序）。
function embed_ref(M::AbstractMatrix, qs::Vector{Int}, n::Int)
    qs0 = qs .- 1                        # 转内部 0-based 位号
    d = 1 << n
    D = 1 << length(qs0)
    out = zeros(ComplexF64, d, d)
    for col in 0:d-1
        lc = 0
        for (j, q) in enumerate(qs0)          # qs0[1] = MSB
            lc = (lc << 1) | ((col >> q) & 1)
        end
        base = col
        for q in qs0
            base &= ~(1 << q)
        end
        for lrow in 0:D-1
            grow = base
            for (j, q) in enumerate(qs0)
                grow |= ((lrow >> (length(qs0) - j)) & 1) << q
            end
            out[grow+1, col+1] = M[lrow+1, lc+1]
        end
    end
    return out
end

@testset "VQC" verbose = true begin
    include("states.jl")
    include("gates.jl")
    include("simulate.jl")
    include("channels.jl")
    include("measure.jl")
    include("ptrace.jl")
    include("expectation.jl")
    include("spinop.jl")
    if Base.find_package("Zygote") === nothing
        @warn "Zygote is not available; AD tests are skipped."
    else
        include("ad.jl")
    end
end
