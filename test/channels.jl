@testset "channels" begin
    # ── 单比特去极化（Pauli 误差语义：ρ' = (1-p)ρ + (p/3)(XρX + YρY + ZρZ)） ──
    # 对 |0⟩：ρ'00 = 1 - 2p/3，ρ'11 = 2p/3
    p = 0.3
    ρ = DensityMatrix(zero_state(1))
    out = apply!(copy(ρ), Depolarizing(1, p))
    @test real(out[1, 1]) ≈ 1 - 2 * p / 3 atol = 1e-12
    @test real(out[2, 2]) ≈ 2 * p / 3 atol = 1e-12
    # 与手工 Kraus 作用对照
    ks = kraus(Depolarizing(1, p).channel)
    expected = zero(storage(ρ))
    for K in ks
        expected .+= K * storage(ρ) * K'
    end
    @test storage(out) ≈ expected atol = 1e-12

    # ── 纯态输入自动转换为 DensityMatrix ──
    out2 = apply!(zero_state(1), Depolarizing(1, p))
    @test out2 isa DensityMatrix
    @test out2 ≈ out

    # ── 振幅阻尼（解析形式） ──
    γ = 0.2
    ψ1 = onehot_encoding([1])
    out3 = apply!(DensityMatrix(ψ1), AmplitudeDamping(1, γ))
    @test real(out3[1, 1]) ≈ γ atol = 1e-12     # |1⟩ 以概率 γ 衰减到 |0⟩
    @test real(out3[2, 2]) ≈ 1 - γ atol = 1e-12

    # 与手工 Kraus 作用对照（一般态）
    ψ = rand_state(2)
    ch = QuantumCircuits.AmplitudeDamping(2, γ).channel
    ks = kraus(ch)
    rho = storage(DensityMatrix(ψ))
    expected4 = zeros(ComplexF64, 4, 4)
    for K in ks
        Ke = embed_ref(K, [2], 2)
        expected4 .+= Ke * rho * Ke'
    end
    @test storage(apply!(DensityMatrix(ψ), QuantumCircuits.AmplitudeDamping(2, γ))) ≈ expected4 atol = 1e-12

    # ── PauliError（QEC） ──
    out4 = apply!(DensityMatrix(zero_state(1)), PauliError(1, (0.1, 0.0, 0.0)))
    # ρ' = 0.9|0⟩⟨0| + 0.1|1⟩⟨1|
    @test real(out4[1, 1]) ≈ 0.9 atol = 1e-12
    @test real(out4[2, 2]) ≈ 0.1 atol = 1e-12

    # ── 相干重置（Kraus: K₀ = |0⟩⟨0|, K₁ = |0⟩⟨1|） ──
    ρx = DensityMatrix(apply!(zero_state(1), X(1)))     # |1⟩
    @test apply!(copy(ρx), QuantumCircuits.reinit(1)) ≈ DensityMatrix(zero_state(1))
    # 直积态重置不影响其余比特
    ψp = qubit_encoding([0.3, 0.8])
    ρp = DensityMatrix(ψp)
    @test apply!(copy(ρp), QuantumCircuits.reinit(1)) ≈ DensityMatrix(qubit_encoding([0.0, 0.8])) atol = 1e-10
    # 纯态相干重置
    ψr = apply!(zero_state(1), X(1))
    @test apply!(copy(ψr), QuantumCircuits.reinit(1)) ≈ zero_state(1)

    # ── 类型语义：以态的 eltype 为准 ──
    # 实算子 + 实态 → 保持实类型；实算子 + 复态 → 保持复类型
    ψre = StateVector(Float64[1.0, 0.0], 1)
    out_re = apply!(copy(ψre), X(1))
    @test eltype(out_re) == Float64
    out_cx = apply!(copy(ψre), Depolarizing(1, p))   # 复 Kraus → 实态升复
    @test eltype(out_cx) <: Complex
    @test out_re ≈ onehot_encoding([1])
    # 实 Kraus + 实态 → 保持实类型
    ks_re = [Matrix{Float64}(I, 2, 2)]               # 单位算子
    ρre = DensityMatrix(Float64[1.0 0.0; 0.0 0.0], 1)
    out_k = apply_kraus!(ρre, ks_re, [1])
    @test eltype(out_k) == Float64
    @test out_k ≈ DensityMatrix(Float64[1.0 0.0; 0.0 0.0], 1)
    # 复 Kraus + 实态 → 升复
    ρre2 = DensityMatrix(Float64[1.0 0.0; 0.0 0.0], 1)
    out_k2 = apply_kraus!(ρre2, [Matrix{ComplexF64}(I, 2, 2)], [1])
    @test eltype(out_k2) <: Complex
end
