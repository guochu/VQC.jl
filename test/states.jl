"最大混合态。"
function maximally_mixed(n::Int)
    d = 1 << n
    return DensityMatrix(Matrix{ComplexF64}(I, d, d) / d, n)
end

@testset "states" begin
    # 构造
    ψ = zero_state(3)
    @test nqubits(ψ) == 3
    @test storage(ψ)[1] == 1
    @test ψ == StateVector(ComplexF64, 3)
    @test StateVector(3) isa StateVector{ComplexF64}

    v = rand_state(2)
    @test norm(v) ≈ 1
    @test nqubits(v) == 2
    @test copy(v) == v

    # 代数
    a = rand_state(2); b = rand_state(2)
    @test (a + b).data ≈ a.data + b.data
    @test (2 * a).data ≈ 2 * a.data
    @test dot(a, b) ≈ dot(a.data, b.data)

    # 编码
    @test onehot_encoding([1, 0]) == StateVector([0.0im, 1, 0, 0], 2)  # qubit0=1
    @test abs2.(storage(qubit_encoding([0.0]))) ≈ [1.0, 0.0]
    @test abs2.(storage(qubit_encoding([1.0]))) ≈ [0.0, 1.0]
    θs = [0.3, -0.7]
    prod_state = qubit_encoding(θs)
    expect00 = cos(π * θs[1] / 2) * cos(π * θs[2] / 2)
    @test real(prod_state[1]) ≈ expect00 atol = 1e-12
    amp = amplitude_encoding([2.0, 0, 0, 0]; nqubits=2)
    @test amp ≈ zero_state(2)

    # 就地重置
    x = rand_state(2)
    reset!(x)
    @test x == zero_state(2)
    reset_onehot!(x, [1, 1])
    @test abs2(x[4]) ≈ 1
    reset_qubit!(x, θs)
    @test x ≈ prod_state

    # permute：小端序，qubit 1 = 最低位（1-based）
    p = onehot_encoding([1, 0])          # |01⟩（qubit 1 = 1）
    @test permute(p, [2, 1]) ≈ onehot_encoding([0, 1])

    # 保真度 / 距离
    @test fidelity(zero_state(2), zero_state(2)) ≈ 1
    @test fidelity(zero_state(2), onehot_encoding([0, 1])) ≈ 0
    @test distance(zero_state(2), zero_state(2)) ≈ 0
    @test distance(zero_state(2), onehot_encoding([0, 1])) ≈ sqrt(2)

    # DensityMatrix
    ρ = DensityMatrix(zero_state(2))
    @test nqubits(ρ) == 2
    @test ρ[1, 1] == 1
    @test tr(ρ) ≈ 1
    ρ2 = DensityMatrix(rand_state(2))
    @test tr(ρ2) ≈ 1 atol = 1e-12
    @test ishermitian(ρ2)
    @test fidelity(ρ, ρ) ≈ 1
    @test ρ ≈ DensityMatrix(zero_state(2))

    rd = rand_densitymatrix(2)
    @test tr(rd) ≈ 1 atol = 1e-12
    @test all(real.(schmidt_numbers(rd)) .>= -1e-12)
    @test renyi_entropy(real.(schmidt_numbers(maximally_mixed(2))); α=2) ≈ 2 atol = 1e-8

    # 纯态密度矩阵 ↔ 保真度
    ψa = rand_state(2)
    @test fidelity(DensityMatrix(ψa), ψa) ≈ 1

    # permute DM
    ρp = DensityMatrix(onehot_encoding([1, 0]))
    @test permute(ρp, [2, 1]) ≈ DensityMatrix(onehot_encoding([0, 1]))
end
