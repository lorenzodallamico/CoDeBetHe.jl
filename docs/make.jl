import Pkg; Pkg.add("Plots")
using Documenter, CoDeBetHe, Plots


makedocs(
    doctest = true,
    format = Documenter.HTML(prettyurls=!("local" in ARGS)),
    sitename = "CoDeBetHe.jl",
    authors = "Lorenzo Dall'Amico, Nicolas Tremblay",
    pages = [
        "CoDeBetHe" => "index.md"
        "Static CD" => "man/static_CD.md"
        "Dynamic CD" => "man/dynamic_CD.md"
        "Useful functions" => "man/useful_functions.md"
        ]
)

deploydocs(
    repo = "github.com/lorenzodallamico/CoDeBetHe.jl.git",
    target = "build",
    push_preview = true,
)
