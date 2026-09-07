@testset "partial trace" begin
    Random.seed!(99)

    # Bell 态：约化到任一比特为最大混合
    c = Circuit(2)
    push!(c, H(1)); push!(c, CX(1, 2))
    ψ = simulate(c, zero_state(2))
    ρ0 = partial_tr(ψ, [2])
    @test ρ0 isa DensityMatrix && nqubits(ρ0) == 1
    @test storage(ρ0) ≈ Matrix{ComplexF64}(I, 2, 2) / 2 atol = 1e-12
    @test partial_tr(ψ, [1]) ≈ ρ0

    # 直积态：偏迹为子块
    ψp = qubit_encoding([0.3, 0.8])
    @test partial_tr(ψp, [2]) ≈ DensityMatrix(qubit_encoding([0.3]))
    @test partial_tr(ψp, [1]) ≈ DensityMatrix(qubit_encoding([0.8]))

    # 与密度矩阵偏迹一致
    ρp = DensityMatrix(ψp)
    @test partial_tr(ρp, [2]) ≈ partial_tr(ψp, [2])
    @test partial_tr(ρp, [1]) ≈ partial_tr(ψp, [1])

    # 随机纯态：SV 偏迹 = DM 偏迹
    ψ3 = rand_state(3)
    ρ3 = DensityMatrix(ψ3)
    @test partial_tr(ψ3, [3]) ≈ partial_tr(ρ3, [3])
    @test partial_tr(ψ3, [1, 3]) ≈ partial_tr(ρ3, [1, 3])

    # 约化态保持厄米、迹 1
    r = partial_tr(ψ3, [1])
    @test tr(r) ≈ 1 atol = 1e-12
    @test ishermitian(r)

    # vararg 便利形式
    @test partial_tr(ψp, 1) ≈ partial_tr(ψp, [1])

    # 全比特 / 空比特
    @test partial_tr(ρp, [1, 2]) ≈ tr(ρp)
    @test partial_tr(ρp, Int[]) ≈ ρp
end
