# spinop.jl — SpinOpTerm / SpinOpSum：构造、apply、expectation、adjoint
# （mat 名与 QuantumCircuits.mat 撞名，此处用 VQC.mat 的局部别名）

const spin_mat = VQC.mat

@testset "spinop" begin
    Random.seed!(1234)

    @testset "构造与代数" begin
        t = SpinOpTerm(0.5, 2 => :Y, 1 => :Z)
        @test t.coeff == 0.5
        @test first.(t.ops) == [1, 2]              # 位置自动排序
        @test (2 * t).coeff == 1.0
        @test (t * 2).coeff == 1.0
        s = t + SpinOpTerm(1.0, 3 => :X)
        @test s isa SpinOpSum && length(s.terms) == 2
        @test (s + t).terms[end] == t
        @test (-t).coeff == -0.5
        h = SpinOpTerm(0.7, 1 => :Z, 2 => :Z)
        @test ishermitian(h)
        @test !ishermitian(SpinOpTerm(1.0, 1 => :P))
        @test adjoint(SpinOpTerm(1.0, 1 => :P)).ops[1][2] == :M
        @test adjoint(SpinOpTerm(0.5 + 0.5im, 1 => :Y)) == SpinOpTerm(0.5 - 0.5im, 1 => :Y)
    end

    @testset "apply 与 mat 对拍" begin
        n = 4
        A = ComplexF64[0.5 0.2im; -0.3 1.2]        # 任意非厄米 2×2 矩阵
        t = SpinOpTerm(0.7, 1 => :Z, 3 => :Y, 4 => A)
        ψ = rand_state(n)
        v_full = spin_mat(t, n) * storage(ψ)
        v_apply = storage(apply(t, ψ))
        @test norm(v_apply - v_full) / norm(v_full) < 1e-12

        # 同位双算子（乘积顺序：A₁ A₂ = 先 A₂ 后 A₁）
        t2 = SpinOpTerm(1.0, 2 => :P, 2 => :M)
        @test norm(storage(apply(t2, ψ)) - spin_mat(t2, n) * storage(ψ)) / norm(storage(ψ)) < 1e-12

        # 任意 2×2 非厄米矩阵（不限 Pauli）
        t3 = SpinOpTerm(1.0, 2 => A)
        @test norm(storage(apply(t3, ψ)) - spin_mat(t3, n) * storage(ψ)) / norm(storage(apply(t3, ψ))) < 1e-12

        # SpinOpSum 的作用
        H = SpinOpSum([SpinOpTerm(0.5, 1 => :Z, 2 => :Z),
                       SpinOpTerm(1.0, 2 => :X),
                       SpinOpTerm(0.3, 1 => :Y, 4 => A)])
        @test norm(storage(apply(H, ψ)) - spin_mat(H, n) * storage(ψ)) / norm(storage(apply(H, ψ))) < 1e-12

        # 位置越界
        @test_throws ArgumentError apply(SpinOpTerm(1.0, 5 => :X), ψ)
        @test_throws ArgumentError SpinOpTerm(1.0, 1 => randn(3, 3))
    end

    @testset "expectation" begin
        n = 4
        ψ = rand_state(n)
        ρ = DensityMatrix(ψ)
        H = SpinOpSum([SpinOpTerm(0.5, 1 => :Z, 2 => :Z),
                       SpinOpTerm(1.0, 2 => :X),
                       SpinOpTerm(0.3 - 0.2im, 1 => :Y, 3 => :X)])
        M = spin_mat(H, n)
        @test expectation(H, ψ) ≈ dot(storage(ψ), M, storage(ψ)) atol = 1e-12
        @test expectation(H, ρ) ≈ sum(vec(storage(ρ)) .* vec(transpose(M))) atol = 1e-12
        @test real(expectation(H, ψ)) ≈ real(expectation(H, ρ)) atol = 1e-12
        # 厄米算符期望为实数
        hH = adjoint(H) + H
        @test imag(expectation(hH, ψ)) ≈ 0 atol = 1e-12
        # 本征态求解场景：|0…0⟩ 是 Z⊗…⊗Z 的 +1 本征态
        g = zero_state(n)
        Hzz = SpinOpSum([SpinOpTerm(1.0, [q => :Z for q in 1:n]...)])
        @test expectation(Hzz, g) ≈ 1.0 atol = 1e-12
    end

    @testset "时间演化（Trotter 步）" begin
        # exp(-i Δt (Z₁Z₂ + X₂)) 用一阶 Trotter 与精确对拍
        n = 2
        Δt = 0.1
        H = SpinOpSum([SpinOpTerm(1.0, 1 => :Z, 2 => :Z), SpinOpTerm(1.0, 2 => :X)])
        ψ = rand_state(n)
        # 精确
        ψ_exact = exp(-1.0im * Δt * spin_mat(H, n)) * storage(ψ)
        # Trotter：逐项 exp(-i Δt term)
        ψ_trot = copy(storage(ψ))
        for term in H.terms
            mt = spin_mat(term, n)
            ψ_trot = exp(-1.0im * Δt * mt) * ψ_trot
        end
        @test norm(ψ_trot - ψ_exact) / norm(ψ_exact) < 1e-2   # 一阶 Trotter 误差 O(Δt²)
    end
end
