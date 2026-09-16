# Runs one example and saves every figure it produces.
#
#   julia --project=docs/figuregen runner.jl <markdown file> <figure dir> <manifest>
#
# The example is executed one block at a time in a dedicated module, so that the
# names it defines cannot collide with the runner's own. A figure is saved when the
# example returns it from a block, displays it, or leaves it in a global variable,
# which between them cover every way the examples produce figures. Blocks that call
# train() are bracketed with markers that generate.jl uses to cut the terminal
# recording down to the training run, and figures from blocks that call plotrisk()
# are named with a "risk" suffix.

include(joinpath(@__DIR__, "config.jl"))
include(joinpath(@__DIR__, "extract.jl"))

using CairoMakie
const Makie = CairoMakie.Makie

const REC_START = "@@FIGGEN_REC_START@@"
const REC_END = "@@FIGGEN_REC_END@@"

length(ARGS) == 3 || error("usage: runner.jl <markdown file> <figure dir> <manifest>")
const MDPATH, FIGDIR, MANIFEST = ARGS
const SLUG = fileslug(MDPATH)

mkpath(FIGDIR)
const manifest = open(MANIFEST, "w")
record!(kind, name) = (println(manifest, "$kind\t$name"); flush(manifest))

# ---------------------------------------------------------------------------
# Figure names
# ---------------------------------------------------------------------------

const name_counts = Dict{String, Int}()

function figure_path(heading::AbstractString, suffix::AbstractString = "", ext::AbstractString = "png")
    section = headingslug(heading)
    stem = isempty(section) ? SLUG : "$(SLUG)_$(section)"
    if !isempty(suffix) && !endswith(stem, suffix)
        stem = "$(stem)_$(suffix)"
    end
    n = get(name_counts, stem, 0) + 1
    name_counts[stem] = n
    filename = (n == 1 ? stem : "$(stem)_$(n)") * "." * ext
    return filename, joinpath(FIGDIR, filename)
end

# ---------------------------------------------------------------------------
# The module the example runs in, and the figures it produces
# ---------------------------------------------------------------------------

const Sandbox = Module(:ExampleSandbox)
const shown_figures = Makie.Figure[]
const saved = Set{UInt}()

# Intercepting display() means `display(fig)` is captured without relying on a
# Makie screen being available, and without the example being modified.
function figgen_display(x)
    x isa Makie.Figure && return (push!(shown_figures, x); nothing)
    return Base.display(x)
end
Core.eval(Sandbox, :(const display = $(figgen_display)))

function sandbox_figures()
    figs = Makie.Figure[]
    for n in names(Sandbox; all = true)
        isdefined(Sandbox, n) || continue
        value = try
            getglobal(Sandbox, n)
        catch
            continue
        end
        value isa Makie.Figure && push!(figs, value)
    end
    return figs
end

function save_figure(fig::Makie.Figure, heading::AbstractString; suffix::AbstractString = "")
    objectid(fig) in saved && return nothing
    push!(saved, objectid(fig))
    filename, path = figure_path(heading, suffix)
    Makie.save(path, fig)
    record!("FIGURE", filename)
    println("  saved $filename")
    flush(stdout)
    return nothing
end

# ---------------------------------------------------------------------------
# Run the example
# ---------------------------------------------------------------------------

const NO_TRAIN = get(ENV, "FIGGEN_NO_TRAIN", "") == "1"

blocks = extract_blocks(MDPATH)
println("Running $(basename(MDPATH)): $(length(blocks)) blocks, " *
        "$(count(b -> b.records, blocks)) recorded")
include_string(Sandbox, "using Random; Random.seed!(1)")

for (n, block) in enumerate(blocks)
    section = isempty(block.heading) ? "(page preamble)" : block.heading
    if NO_TRAIN && block.records
        println("Stopping before block $n ($section), which calls train()")
        break
    end
    println("[$n/$(length(blocks))] $section")
    flush(stdout)

    empty!(shown_figures)
    before = Set(objectid.(sandbox_figures()))

    if block.records
        filename, path = figure_path(block.heading, "training", "gif")
        record!("RECORDING", filename)
        println("\n$REC_START")
        flush(stdout)
    end

    value = try
        include_string(Sandbox, block.code, "$(basename(MDPATH)):$(block.line)")
    catch err
        block.records && (println("\n$REC_END"); flush(stdout))
        println(stderr, "\nFailed in block $n ($section), $(basename(MDPATH)) line $(block.line):")
        println(stderr, block.code)
        record!("ERROR", "block $n at line $(block.line)")
        close(manifest)
        showerror(stderr, err, catch_backtrace())
        exit(1)
    end

    if block.records
        println("\n$REC_END")
        flush(stdout)
    end

    # Displayed first, then returned, then anything new left in a global.
    suffix = block.plots_risk ? "risk" : ""
    for fig in shown_figures
        save_figure(fig, block.heading; suffix)
    end
    value isa Makie.Figure && save_figure(value, block.heading; suffix)
    for fig in sandbox_figures()
        objectid(fig) in before || save_figure(fig, block.heading; suffix)
    end
end

record!("DONE", basename(MDPATH))
close(manifest)
println("Finished $(basename(MDPATH))")
