@testset "measurement" begin
    Random.seed!(2024)

    # ── probabilities ──
    ψ = rand_state(3)
    ps = probabilities(ψ)
    @test sum(ps) ≈ 1 atol = 1e-12
    @test ps ≈ abs2.(storage(ψ))
    # 边缘分布：qubit 1（最低位）的 P(1) = 全分布中偶数索引（1-based 奇数位）之和
    marg = marginal_probabilities(ψ, [1])
    @test sum(marg) ≈ 1 atol = 1e-12
    @test marg[2] ≈ sum(abs2.(storage(ψ))[2:2:end]) atol = 1e-12

    ρ = DensityMatrix(ψ)
    @test probabilities(ρ) ≈ ps atol = 1e-10

    # ── measure!（就地坍缩，只返回测量结果） ──
    x = copy(ψ)
    outcome = measure!(x, 1)
    @test outcome in (0, 1)
    @test norm(x) ≈ 1
    @test marginal_probabilities(x, 1)[outcome + 1] ≈ 1 atol = 1e-12

    # 固化行为：测量后态**以概率 1** 处于坍缩后的状态——
    # 非 outcome 子空间的振幅被严格清零
    x0 = copy(ψ)
    o = measure!(x0, 1)                    # 测量 qubit 1（位 0）
    kept = [i for i in 1:length(x0) if ((i - 1) & 1) == o]
    @test storage(x0)[kept] ≈ storage(ψ)[kept] / sqrt(sum(abs2, storage(ψ)[kept])) atol = 1e-12
    @test all(abs(storage(x0)[i]) == 0 for i in setdiff(1:length(x0), kept))

    # 测量多比特（就地坍缩）
    xm = copy(ψ)
    outs = measure!(xm, [1, 3])
    @test outs isa Vector{Int} && length(outs) == 2
    for (t, o) in enumerate(outs)
        @test marginal_probabilities(xm, [1, 3][t])[o + 1] ≈ 1 atol = 1e-12
    end

    # 确定性测量
    y = onehot_encoding([1])
    o0 = measure!(y, 1)
    @test o0 == 1
    @test y ≈ onehot_encoding([1])

    # ── measure!(s)：测量全部比特，返回小端序整数 ──
    za = onehot_encoding([1, 0])          # q1 = 1
    o_all = measure!(za)
    @test o_all == 1
    @test za ≈ onehot_encoding([1, 0])
    zb = rand_state(3)
    o_b = measure!(zb)
    # 坍缩到 |o_b⟩（相位保留，比较模长）
    @test abs.(storage(zb)) ≈ abs.(storage(onehot_encoding([o_b & 1, (o_b >> 1) & 1, (o_b >> 2) & 1])))
    @test probabilities(zb)[o_b + 1] ≈ 1 atol = 1e-12

    # ── measure（非就地） ──
    z = rand_state(2)
    z2, o2, p2 = measure(z, 1)
    @test z2 !== z
    @test p2 ≈ marginal_probabilities(z, 1)[o2 + 1] atol = 1e-12

    # ── sample ──
    ψb = simulate(begin
        cc = Circuit(2)
        push!(cc, H(1)); push!(cc, CX(1, 2)); cc
    end, zero_state(2))
    counts = sample(ψb, 1000)
    @test sum(values(counts)) == 1000
    @test Set(keys(counts)) ⊆ Set([0, 3])       # 只出现 |00⟩ / |11⟩

    counts_sub = sample(ψb, [1], 500)
    @test sum(values(counts_sub)) == 500

    # ── post_select ──
    x = copy(ψb)
    p = post_select!(x, 1, 1)
    @test p ≈ 0.5 atol = 1e-12
    @test norm(x) ≈ 1
    # 后选择 qubit1=1 后状态为 |11⟩
    @test abs2(x[4]) ≈ 1 atol = 1e-12

    x2, p2 = post_select(ψb, 1, 0)
    @test p2 ≈ 0.5 atol = 1e-12
    @test x2 ≈ zero_state(2)

    # 密度矩阵后选择
    rb, _ = post_select(DensityMatrix(ψb), 1, 1)
    @test tr(rb) ≈ 1 atol = 1e-12
    @test real(rb[4, 4]) ≈ 1 atol = 1e-12

    # 概率为零的后选择抛错
    @test_throws ArgumentError post_select!(zero_state(1), 1, 1)
end
