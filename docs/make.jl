using Documenter
using QuantumCircuits
using VQC

makedocs(
	sitename="VQC.jl",
	authors = "Guo Chu",
	pages=["Home" => "index.md",
	"gettingstarted.md",
	"coreapi.md",
	"variational.md",
	"ham.md",
	"qctrl.md"],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    )
	)
