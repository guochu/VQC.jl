using Documenter
using QuantumCircuits
using QuantumCircuits.Hamiltonian: SpinOpTerm, SpinOpSum
using VQC

# 源文件根为 docs/ 本身：普通页面在 docs/src/，教程在 docs/tutorials/。
makedocs(
    root = @__DIR__,
    source = ".",
    sitename = "VQC.jl",
    authors = "Guo Chu",
    pages = ["Home" => "src/index.md",
             "src/gettingstarted.md",
             "src/coreapi.md",
             "src/variational.md",
             "src/ham.md",
             "src/qctrl.md",
             "教程" => [
                 "海森堡基态（VQA）" => "tutorials/vqc/heisenberg.md",
             ]],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
)
