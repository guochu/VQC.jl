@testset "expectation" begin
    Random.seed!(7)
    ψ = rand_state(3)
    ρ = DensityMatrix(ψ)

    # ── PauliTerm：与稠密矩阵对照 ──
    t = PauliTerm(0.7, 1 => :X, 2 => :Z)
    M = QuantumCircuits.Hamiltonian.mat(t, 3)
    @test expectation(t, ψ) ≈ dot(storage(ψ), M, storage(ψ)) atol = 1e-12
    @test expectation(t, ρ) ≈ tr(storage(ρ) * M) atol = 1e-12

    # 恒等项
    ti = PauliTerm(2.5)
    @test expectation(ti, ψ) ≈ 2.5 atol = 1e-12
    @test expectation(ti, ρ) ≈ 2.5 atol = 1e-12

    # 单 Z
    tz = PauliTerm(1.0, 1 => :Z)
    Mz = QuantumCircuits.Hamiltonian.mat(tz, 3)
    @test expectation(tz, ρ) ≈ tr(storage(ρ) * Mz) atol = 1e-12

    # ── PauliSum：与稠密矩阵对照 ──
    h = PauliSum([PauliTerm(0.5, 1 => :Z),
                  PauliTerm(0.3, 2 => :X, 3 => :Y),
                  PauliTerm(-1.2)])
    Mh = mat(h, 3)
    @test expectation(h, ψ) ≈ dot(storage(ψ), Mh, storage(ψ)) atol = 1e-12
    @test expectation(h, ρ) ≈ tr(storage(ρ) * Mh) atol = 1e-12

    # ── GateOp 与一般矩阵当可观测量 ──
    @test expectation(H(1), ψ) ≈ dot(storage(ψ), embed_ref(mat(H), [1], 3), storage(ψ)) atol = 1e-12
    @test expectation(Mh, ρ) ≈ tr(storage(ρ) * Mh) atol = 1e-10
    @test expectation(ψ, Mh, ψ) ≈ dot(storage(ψ), Mh, storage(ψ)) atol = 1e-12

    # 厄米算符的期望应为实数
    @test abs(imag(expectation(h, ψ))) < 1e-12
end
