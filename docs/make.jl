using Documenter, LuxNLPModels

makedocs(
  modules = [LuxNLPModels],
  doctest = true,
  strict = true,
  format = Documenter.HTML(
    assets = ["assets/style.css"],
    prettyurls = get(ENV, "CI", nothing) == "true",
  ),
  sitename = "LuxNLPModels.jl",
  pages = Any["Home" => "index.md", "Tutorial" => "tutorial.md", "Reference" => "reference.md"],
)

deploydocs(
  repo = "github.com/Farhad-Phd/LuxNLPModels.jl.git",
  push_preview = true,
  devbranch = "main",
)
