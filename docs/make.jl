push!(LOAD_PATH,"../src/")

using Documenter, CoDeBetHe


makedocs(
    doctest = true,
    format = Documenter.HTML(prettyurls=!("local" in ARGS)),
    sitename = "CoDeBetHe",
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
)
