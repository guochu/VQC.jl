using Zygote
using QuantumCircuits: Param

@testset "AD" begin
    Random.seed!(4242)

    hZ0 = PauliSum([PauliTerm(1.0, 1 => :Z)])
    hZ0Z1 = PauliSum([PauliTerm(1.0, 1 => :Z), PauliTerm(0.5, 2 => :Z)])

    # ── 1. 态矢量：RX 参数梯度 vs 解析值 ──
    c = Circuit(2)
    push!(c, RX(:θ, 1))
    f = θ -> real(expectation(hZ0, simulate(c, zero_state(2); params=[θ])))
    θ0 = 0.3
    g = Zygote.gradient(f, θ0)[1]
    @test g ≈ -sin(θ0) atol = 1e-9

    # ── 2. 多层线路 + 权重共享：梯度累加 ──
    c2 = Circuit(2)
    push!(c2, RX(:θ, 1))
    push!(c2, RY(:θ, 2))
    f2 = θ -> real(expectation(hZ0Z1, simulate(c2, zero_state(2); params=[θ])))
    g2 = Zygote.gradient(f2, θ0)[1]
    @test g2 ≈ -sin(θ0) - 0.5 * sin(θ0) atol = 1e-9

    # ── 3. 一般参数门（自定义 2-qubit ParamGate，非 Rx/Ry/Rz）──
    ugate = QuantumCircuits.ParamGate(:myrot, 2, 2, (θ, φ) -> begin
        c, s = cos(θ / 2), sin(θ / 2)
        ComplexF64[c 0 0 -im*s*cos(φ); 0 c -im*s*sin(φ) 0;
                   0 -im*s*sin(φ) c 0; -im*s*cos(φ) 0 0 c]
    end)
    c3 = Circuit(2)
    push!(c3, GateOp(ugate, [1, 2], [Param(:θ), Param(:φ)]))
    f3 = (θ, φ) -> begin
        s = simulate(c3, zero_state(2); params=[θ, φ])
        return real(expectation(hZ0Z1, s))
    end
    θv, φv = 0.4, -0.6
    g3 = Zygote.gradient(f3, θv, φv)
    fd3(θ, φ) = (f3(θ + 1e-7, φ) - f3(θ - 1e-7, φ)) / 2e-7
    @test g3[1] ≈ fd3(θv, φv) atol = 1e-6
    fdφ(θ, φ) = (f3(θ, φ + 1e-7) - f3(θ, φ - 1e-7)) / 2e-7
    @test g3[2] ≈ fdφ(θv, φv) atol = 1e-6

    # ── 4. 受控参数门：ctrl(RX(:θ,2), 1) ──
    c4 = Circuit(2)
    push!(c4, X(1))
    push!(c4, ctrl(RX(:θ, 2), 1))
    f4 = θ -> real(expectation(hZ0, simulate(c4, zero_state(2); params=[θ])))
    g4 = Zygote.gradient(f4, θ0)[1]
    fd4(θ) = (f4(θ + 1e-7) - f4(θ - 1e-7)) / 2e-7
    @test g4 ≈ fd4(θ0) atol = 1e-6

    # ── 5. 密度矩阵（含噪线路）：解析梯度 vs AD ──
    # |0⟩ —RX(θ)— AmplitudeDamping(γ):  ⟨Z⟩ = γ + (1−γ)cosθ
    γ = 0.1
    c5 = Circuit(1)
    push!(c5, RX(:θ, 1))
    push!(c5, AmplitudeDamping(1, γ))
    f5 = θ -> begin
        rho = simulate(c5, DensityMatrix(zero_state(1)); params=[θ])
        return real(expectation(hZ0, rho))
    end
    @test f5(θ0) ≈ γ + (1 - γ) * cos(θ0) atol = 1e-12
    g5 = Zygote.gradient(f5, θ0)[1]
    @test g5 ≈ -(1 - γ) * sin(θ0) atol = 1e-9

    # ── 6. 去极化信道梯度 vs 差分 ──
    c6 = Circuit(1)
    push!(c6, RX(:θ, 1))
    push!(c6, Depolarizing(1, 0.05))
    f6 = θ -> begin
        rho = simulate(c6, DensityMatrix(zero_state(1)); params=[θ])
        return real(expectation(hZ0, rho))
    end
    g6 = Zygote.gradient(f6, θ0)[1]
    fd6(θ) = (f6(θ + 1e-7) - f6(θ - 1e-7)) / 2e-7
    @test g6 ≈ fd6(θ0) atol = 1e-6

    # ── 7. 初始态梯度（实参数直积编码，Zygote 原生追踪构造）──
    c7 = Circuit(2)
    push!(c7, RZ(0.8, 1))
    f7 = θs -> begin
        ψ = qubit_encoding(θs)
        real(expectation(hZ0Z1, simulate(c7, ψ)))
    end
    θs0 = [0.3, 0.9]
    g7 = Zygote.gradient(f7, θs0)[1]
    fd7 = [let δ = [i == 1 ? 1e-7 : 0.0, i == 2 ? 1e-7 : 0.0]
               (f7(θs0 .+ δ) - f7(θs0 .- δ)) / 2e-7
           end for i in 1:2]
    @test g7 ≈ fd7 atol = 1e-6

    # ── 8. 恒等项参与的 PauliSum 伴随（DM 上）──
    h8 = PauliSum([PauliTerm(0.7, 1 => :Z), PauliTerm(1.0)])
    f8 = θ -> begin
        rho = simulate(c5, DensityMatrix(zero_state(1)); params=[θ])
        return real(expectation(h8, rho))
    end
    g8 = Zygote.gradient(f8, θ0)[1]
    fd8(θ) = (f8(θ + 1e-7) - f8(θ - 1e-7)) / 2e-7
    @test g8 ≈ fd8(θ0) atol = 1e-6

    # ── 9. post_select 梯度 vs 差分 ──
    c9b = Circuit(2)
    push!(c9b, RX(:θ, 1))
    push!(c9b, RY(π / 4, 2))
    f9b = θ -> begin
        s2, p = post_select(simulate(c9b, zero_state(2); params=[θ]), 1, 1)
        return p * real(expectation(hZ0, s2))
    end
    g9 = Zygote.gradient(f9b, θ0)[1]
    fd9(θ) = (f9b(θ + 1e-7) - f9b(θ - 1e-7)) / 2e-7
    @test g9 ≈ fd9(θ0) atol = 1e-6

    # ── 10. BlockOp 透明回传 ──
    c11 = Circuit(2)
    body = Circuit(1)
    push!(body, RX(:θ, 1))
    push!(c11, QuantumCircuits.block(body; name=:layer, at=[1]))
    f11 = θ -> real(expectation(hZ0, simulate(c11, zero_state(2); params=[θ])))
    g11 = Zygote.gradient(f11, θ0)[1]
    @test g11 ≈ -sin(θ0) atol = 1e-9

    # ── 12. 不可微指令给出明确错误 ──
    c12 = Circuit(2)
    push!(c12, RX(:θ, 1))
    push!(c12, measure(1, c12.cregs[1][1]))
    f12 = θ -> real(expectation(hZ0, simulate(c12, zero_state(2); params=[θ])))
    @test_throws ArgumentError Zygote.gradient(f12, θ0)

    # ── 13. 含噪线路须用 DensityMatrix：给出明确错误 ──
    c13 = Circuit(1)
    push!(c13, RX(:θ, 1))
    push!(c13, Depolarizing(1, 0.1))
    f13 = θ -> begin
        s = simulate(c13, zero_state(1); params=[θ])
        return real(expectation(hZ0, s))
    end
    @test_throws ArgumentError Zygote.gradient(f13, θ0)
end
