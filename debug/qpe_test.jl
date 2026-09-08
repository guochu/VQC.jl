# debug/qpe_test.jl — QPE 线路验证
#
# U = PHASE(θ) 作用在特征比特 |1⟩ 上，本征值 e^{iθ}。
# m = 4 个寻址比特的 QPE 应以概率 1 测得 y = M·θ/(2π) = 16·(π/4)/(2π) = 2。

using QuantumCircuits, VQC

"2-qubit 受控相位门 CP(φ) = diag(1, 1, 1, e^{iφ})"
cp_gate(φ) = QuantumCircuits.usergate(:cp,
    [1.0 0 0 0; 0 1.0 0 0; 0 0 1.0 0; 0 0 0 cis(φ)])

# 逆 QFT（LSB-first 位号 qs[j] 对应整数 y 的位 j-1）
function inverse_qft!(c, qs::NTuple{M,Int}) where {M}
    for j in M:-1:1                       # 从最高位往回解
        push!(c, H(qs[j]))
        for k in 1:j-1                    # 控制位 qs[k]（更低位），相位作用于 qs[j]
            push!(c, cp_gate(-π / 2^(j - k))(qs[k], qs[j]))
        end
    end
    return c
end

function qpe_circuit(θ::Real; M::Int = 4)
    c = Circuit(M + 1)
    addr = ntuple(j -> j, M)              # 寻址位 (1,2,…,M)
    feat = M + 1                          # 特征位
    push!(c, X(feat))                     # 特征态 |1⟩
    for j in 1:M
        push!(c, H(addr[j]))
    end
    for j in 1:M                          # 受控 U^{2^{j-1}}：控制 addr[j]
        push!(c, cp_gate(2^(j - 1) * θ)(addr[j], feat))
    end
    inverse_qft!(c, addr)
    return c, addr
end

θ = π / 4
M = 4
c, addr = qpe_circuit(θ)
counts = Dict{Int,Int}()
for _ in 1:200
    ψ = simulate(c, zero_state(M + 1))
    y = measure!(ψ, collect(1:M))          # 寻址位结果向量
    # iQFT 输出为位反序：k 按 MSB-first 组装（addr[1] = 最高位）
    k = sum((y[j] << (M - j)) for j in 1:M)
    counts[k] = get(counts, k, 0) + 1
end
println("θ = π/4, 精确解 k = 2^M·θ/(2π) = ", 2^M * θ / (2π))
println("测量分布（k → 次数）: ", sort(collect(counts)))
println("θ̂ = 2π·k/2^M = ", 2π * (argmax(counts) |> first) / 2^M, "（弧度）")
