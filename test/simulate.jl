@testset "simulate" begin
    # Bell 态
    c = Circuit(2)
    push!(c, H(1))
    push!(c, CX(1, 2))
    ψ = simulate(c, zero_state(2))
    @test probabilities(ψ) ≈ [0.5, 0, 0, 0.5] atol = 1e-12

    # 与稠密矩阵组合一致
    M = embed_ref(mat(CX), [1, 2], 2) * embed_ref(mat(H), [1], 2)
    ref = M * storage(zero_state(2))
    @test storage(ψ) ≈ ref

    # ── 符号参数：simulate(params) 与 assign 等价 ──
    cp = Circuit(2)
    push!(cp, RX(:θ, 1))
    push!(cp, RY(:θ, 2))                     # 同名参数 = 权重共享
    push!(cp, CX(1, 2))
    θv = 0.4
    s1 = simulate(cp, zero_state(2); params=[θv])
    c2 = QuantumCircuits.assign(copy(cp), Dict(:θ => θv))
    s2 = simulate(c2, zero_state(2))
    @test s1 ≈ s2

    # 参数表（Dict）
    s3 = simulate(cp, zero_state(2); params=Dict(:θ => θv))
    @test s3 ≈ s1

    # ── 测量 + 经典条件分支 ──
    cm = Circuit(1)
    push!(cm, H(1))
    push!(cm, measure(1, cm.cregs[1][1]))
    QuantumCircuits.if_then(cm, cm.cregs[1][1] == 1, Circuit([X(1)]))
    out = simulate(cm, zero_state(1))
    @test probabilities(out)[1] ≈ 1 atol = 1e-12   # 测到 1 会被 X 翻回 |0⟩

    # 单比特条件（index ≥ 2）：测量写 c2[2]，条件 c2[2]==1 翻转 qubit 2
    cm2 = Circuit(2; cregs = [QuantumCircuits.CReg(:c2, 2)])
    push!(cm2, H(1))
    push!(cm2, measure(1, cm2.cregs[1][2]))
    QuantumCircuits.if_then(cm2, cm2.cregs[1][2] == 1, Circuit([X(2)]))
    p1 = 0.0
    for _ in 1:200
        out2 = simulate(cm2, zero_state(2))
        p1 += marginal_probabilities(out2, [2])[2]
    end
    @test 0.4 < p1 / 200 < 0.6   # 条件按位触发（回归：索引不得丢失）

    # ── BlockOp（重复 + 映射）──
    body = Circuit([X(1)]; n=1)
    cb = Circuit(4)
    push!(cb, QuantumCircuits.block(body; name=:flip, at=[3]))
    outb = simulate(cb, zero_state(4))
    @test probabilities(outb)[(1 << 2) + 1] ≈ 1 atol = 1e-12   # 1-based qubit 3

    # repeat=2 相互抵消
    cb2 = Circuit(2)
    push!(cb2, QuantumCircuits.block(Circuit([X(1)]; n=1); at=[1], repeat=2))
    @test simulate(cb2, zero_state(2)) ≈ zero_state(2)

    # ── ReinitOp / BarrierOp ──
    cr = Circuit(2)
    push!(cr, X(2))
    push!(cr, QuantumCircuits.reinit(2))
    @test simulate(cr, zero_state(2)) ≈ zero_state(2)

    cbar = Circuit(2)
    push!(cbar, barrier(1, 2))
    @test simulate(cbar, zero_state(2)) ≈ zero_state(2)

    # ── simulate! 就地 ──
    x = zero_state(2)
    simulate!(c, x)
    @test x ≈ ψ
end
