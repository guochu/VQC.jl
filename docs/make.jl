push!(LOAD_PATH, "../src")

using Documenter
using VQC

makedocs(
	sitename="VQC.jl",
	authors = "Guo Chu",
	pages=["Home" => "index.md",
	"gettingstarted.md",
	"variational.md",
	"ham.md",
	"qctrl.md"],
	format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    )
	)