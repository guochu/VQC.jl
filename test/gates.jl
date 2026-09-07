using Random

@testset "gates" begin
    Random.seed!(1234)
    n = 4

    # ── 随机 2-qubit 酉门作用于随机态（MSB-first 约定）──
    U = Matrix{ComplexF64}(qr(randn(ComplexF64, 4, 4)).Q)
    g = QuantumCircuits.usergate(:u2, U)
    qs = [3, 1]                          # qubit 3 = 矩阵最高位
    ψ = rand_state(n)
    expected = embed_ref(U, qs, n) * storage(ψ)
    @test apply!(copy(ψ), g(qs...)) ≈ StateVector(expected, n)

    # ── 密度矩阵上的酉演化 ──
    ρ = DensityMatrix(rand_state(n))
    expected_ρ = embed_ref(U, qs, n) * storage(ρ) * embed_ref(U, qs, n)'
    @test apply!(copy(ρ), g(qs...)) ≈ DensityMatrix(expected_ρ, n)

    # ── 3-qubit 门（静态路径）──
    U3 = Matrix{ComplexF64}(qr(randn(ComplexF64, 8, 8)).Q)
    g3 = QuantumCircuits.usergate(:u3, U3)
    qs3 = [4, 2, 3]
    ψ3 = rand_state(4)
    @test apply!(copy(ψ3), g3(qs3...)) ≈ StateVector(embed_ref(U3, qs3, 4) * storage(ψ3), 4)

    # ── 5-qubit 门（动态路径 N > 4）──
    U5 = Matrix{ComplexF64}(qr(randn(ComplexF64, 32, 32)).Q)
    g5 = QuantumCircuits.usergate(:u5, U5)
    qs5 = [2, 4, 1, 6, 3]
    ψ5 = rand_state(6)
    @test apply!(copy(ψ5), g5(qs5...)) ≈ StateVector(embed_ref(U5, qs5, 6) * storage(ψ5), 6)

    # ── 标准库门：CX（控制比特在列表首位 = 矩阵最高位）──
    # ψ = |q2=0, q1=1⟩：控制 qubit 1 = 1 → 目标 qubit 2 翻转
    ψ10 = onehot_encoding([1, 0])        # qubit 1 = 1
    @test apply!(copy(ψ10), CX(1, 2)) ≈ onehot_encoding([1, 1])
    # 控制 qubit 1 = 0 → 不翻转
    @test apply!(copy(onehot_encoding([0, 0])), CX(1, 2)) ≈ onehot_encoding([0, 0])

    # ── 修饰门：ctrl / inv / pow ──
    # ctrl(H(1), 2)：qubit 2 为控制（列表首位 = 最高位）
    op = ctrl(H(1), 2)
    ψc = rand_state(2)
    expected_c = embed_ref(QuantumCircuits.mat(op.gate), [2, 1], 2) * storage(ψc)
    @test apply!(copy(ψc), op) ≈ StateVector(expected_c, 2)

    # 逆门
    θ = 0.7
    ψr = rand_state(3)
    fwd = apply!(copy(ψr), RX(θ, 3))
    back = apply!(fwd, inv(RX(θ, 3)))
    @test back ≈ ψr

    # 幂门：pow(H, 2) = I
    @test apply!(copy(ψc), pow(H(1), 2)) ≈ ψc

    # SWAP
    ψ01 = onehot_encoding([1, 0])
    @test apply!(copy(ψ01), SWAP(1, 2)) ≈ onehot_encoding([0, 1])

    # 参数门（数值参数）
    ψz = zero_state(2)
    @test probabilities(apply!(copy(ψz), RY(π, 1)))[2] ≈ 1 atol = 1e-12

    # 实数态作用复数门时自动提升
    re = StateVector(Float64[1.0, 0.0], 1)
    out = apply!(copy(re), RX(0.5, 1))
    @test eltype(out) <: Complex
    @test out ≈ apply!(copy(zero_state(1)), RX(0.5, 1))
end
