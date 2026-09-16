# Drives the example figure pipeline. Normally invoked through run.sh.
#
#   julia --project=docs/figuregen generate.jl [options] [example.md ...]
#
# Options:
#   --extract-only   write the extracted scripts and stop
#   --no-train       stop each example before its first train() call
#   --no-gif         run the examples without recording the terminal
#   --gif-only       redo the GIFs from an earlier run's recordings, running nothing
#   --figdir PATH    where to write figures (default docs/src/assets/figures)

include(joinpath(@__DIR__, "config.jl"))
include(joinpath(@__DIR__, "extract.jl"))

const REC_START = "@@FIGGEN_REC_START@@"
const REC_END = "@@FIGGEN_REC_END@@"

const FIGUREGEN = @__DIR__
const DOCS = dirname(FIGUREGEN)
const EXAMPLES = joinpath(DOCS, "src", "examples")
const GENERATED = joinpath(FIGUREGEN, "generated")

find_tool(name) = something(Sys.which(name), let p = joinpath(homedir(), ".local", "bin", name)
        isfile(p) ? p : nothing
    end, Some(nothing))

# ---------------------------------------------------------------------------
# Turning a recording into a GIF
# ---------------------------------------------------------------------------

event_time(line) = parse(Float64, line[2:(findfirst(',', line) - 1)])
set_event_time(line, t) = "[" * string(round(t, digits = 6)) * line[findfirst(',', line):end]

"""
    trim_cast(castpath, index, outpath)

Write the `index`th recorded training run from `castpath` to `outpath`, rebasing
its timestamps to zero. A run longer than `GIF_TARGET_SECONDS` has its timestamps
scaled down to that length, which keeps every epoch in the GIF while bounding its
size. Shorter runs are left at their recorded pace rather than stretched.
"""
function trim_cast(castpath::AbstractString, index::Integer, outpath::AbstractString)
    lines = readlines(castpath)
    isempty(lines) && return false
    header, events = lines[1], lines[2:end]

    starts = findall(l -> occursin(REC_START, l), events)
    stops = findall(l -> occursin(REC_END, l), events)
    (index <= length(starts) && index <= length(stops)) || return false

    window = events[(starts[index] + 1):(stops[index] - 1)]
    isempty(window) && return false

    t0 = event_time(first(window))
    times = [event_time(l) - t0 for l in window]
    total = last(times)

    scale = total > GIF_TARGET_SECONDS ? GIF_TARGET_SECONDS / total : 1.0
    kept = [set_event_time(l, t * scale) for (l, t) in zip(window, times)]

    open(outpath, "w") do io
        println(io, header)
        foreach(l -> println(io, l), kept)
    end
    return true
end

const _ANSI = r"\e\[[0-9;?]*[a-zA-Z]|\e\][^\a]*\a|\e[()][A-Z0-9]"

function unescape_cast_text(field::AbstractString)
    out = IOBuffer()
    i = firstindex(field)
    while i <= lastindex(field)
        c = field[i]
        if c == '\\' && i < lastindex(field)
            nxt = field[i + 1]
            if nxt == 'n'
                print(out, '\n')
            elseif nxt == 'r'
                print(out, '\r')
            elseif nxt == 't'
                print(out, '\t')
            elseif nxt == 'b'
                print(out, '\b')
            elseif nxt == 'u' && i + 5 <= lastindex(field)
                print(out, Char(parse(UInt16, field[(i + 2):(i + 5)]; base = 16)))
                i += 6
                continue
            else
                print(out, nxt)
            end
            i += 2
        else
            print(out, c)
            i = nextind(field, i)
        end
    end
    return String(take!(out))
end

"""
    still_from_cast(castpath, outpath)

Fallback for when agg is unavailable: render the final frame of a recording to a
PNG with ImageMagick.
"""
function still_from_cast(castpath::AbstractString, outpath::AbstractString)
    convert_exe = find_tool("magick")
    isnothing(convert_exe) && (convert_exe = find_tool("convert"))
    isnothing(convert_exe) && return false

    text = IOBuffer()
    for line in Iterators.drop(readlines(castpath), 1)
        fields = split(line, "\"o\",\"", limit = 2)
        length(fields) == 2 || continue
        print(text, unescape_cast_text(chop(fields[2], tail = 2)))
    end

    # Carriage returns and cursor movement overwrite lines in place; keeping the
    # last version of each is a good enough approximation of the final frame.
    plain = replace(String(take!(text)), _ANSI => "")
    rendered = String[]
    for chunk in eachsplit(plain, '\n')
        push!(rendered, String(last(collect(eachsplit(chunk, '\r')))))
    end
    filter!(!isempty, rendered)
    tail = rendered[max(1, end - GIF_ROWS + 1):end]

    textfile = tempname()
    write(textfile, join(tail, "\n"))
    try
        run(pipeline(`$convert_exe -background "#121314" -fill "#d0d0d0"
                      -font DejaVu-Sans-Mono -pointsize 18 label:@$textfile $outpath`,
            stderr = devnull))
    catch
        run(pipeline(`$convert_exe -background "#121314" -fill "#d0d0d0"
                      -pointsize 18 label:@$textfile $outpath`, stderr = devnull))
    end
    return isfile(outpath)
end

# ---------------------------------------------------------------------------
# Running one example
# ---------------------------------------------------------------------------

read_manifest(path) = isfile(path) ?
                      [Tuple(split(l, '\t')) for l in readlines(path) if occursin('\t', l)] :
                      Tuple{SubString{String}, SubString{String}}[]

"""
    write_gifs(cast, recordings, figdir)

Cut `cast` down to each recorded training run and convert it to a GIF, falling
back to a still PNG when agg is unavailable.
"""
function write_gifs(cast::AbstractString, recordings, figdir::AbstractString)
    agg = find_tool("agg")
    for (k, name) in enumerate(recordings)
        trimmed = joinpath(GENERATED, "$(splitext(name)[1]).cast")
        trim_cast(cast, k, trimmed) || (println(stderr, "  no recording found for $name"); continue)
        if isnothing(agg)
            still = joinpath(figdir, splitext(name)[1] * ".png")
            still_from_cast(trimmed, still) ?
            println("  agg not found: wrote $(basename(still)) instead of $name") :
            println(stderr, "  could not render $name")
        else
            # trim_cast has already set the playback length, so --speed stays at 1.
            run(`$agg --rows $GIF_ROWS --font-size $GIF_FONT_SIZE --fps-cap $GIF_FPS_CAP
                 --idle-time-limit 1 --last-frame-duration 3
                 $trimmed $(joinpath(figdir, name))`)
            println("  wrote $name")
        end
    end
    return nothing
end

"""
    rebuild_gifs(mdpath; figdir)

Redo the GIFs for one example from the recording of an earlier run, without
running anything. The full cast and the manifest are kept in `GENERATED`, so the
appearance of a GIF can be changed without retraining.
"""
function rebuild_gifs(mdpath::AbstractString; figdir::AbstractString)
    slug = fileslug(mdpath)
    cast = joinpath(GENERATED, "$slug.cast")
    entries = read_manifest(joinpath(GENERATED, "$slug.manifest"))
    recordings = [e[2] for e in entries if e[1] == "RECORDING"]
    println("\n", "="^78)
    println("$(basename(mdpath)): $(length(recordings)) recordings")
    if !isfile(cast) || isempty(recordings)
        println(stderr, "  no recording from an earlier run: run the example first")
        return false
    end
    write_gifs(cast, recordings, figdir)
    return true
end

function run_example(mdpath::AbstractString; figdir::AbstractString, extract_only::Bool, gif::Bool)
    slug = fileslug(mdpath)
    blocks = extract_blocks(mdpath)
    mkpath(GENERATED)
    script = joinpath(GENERATED, "$slug.jl")
    write(script, script_text(mdpath, blocks))
    println("\n", "="^78)
    println("$(basename(mdpath)): $(length(blocks)) blocks -> $(relpath(script))")
    extract_only && return true

    manifest = joinpath(GENERATED, "$slug.manifest")
    rm(manifest; force = true)
    julia = joinpath(Sys.BINDIR, "julia")
    runner = joinpath(FIGUREGEN, "runner.jl")
    inner = `$julia --project=$FIGUREGEN --threads=auto $runner $mdpath $figdir $manifest`

    asciinema = gif ? find_tool("asciinema") : nothing
    cast = joinpath(GENERATED, "$slug.cast")
    ok = true
    if isnothing(asciinema)
        gif && println("asciinema not found: running without a recording")
        ok = success(run(ignorestatus(inner)))
    else
        ENV["ASCIINEMA_CONFIG_HOME"] = joinpath(GENERATED, ".asciinema")
        ENV["TERM"] = "xterm-256color"   # recorded in the cast, so pin it
        command = "stty cols $TERM_COLS rows $TERM_ROWS 2>/dev/null; " * string(inner)[2:(end - 1)]
        rec = `$asciinema rec --overwrite --quiet --idle-time-limit $IDLE_TIME_LIMIT
               --cols $TERM_COLS --rows $TERM_ROWS -c $command $cast`
        ok = success(run(ignorestatus(rec)))
    end

    entries = read_manifest(manifest)
    if !ok || any(e -> e[1] == "ERROR", entries)
        failure = findfirst(e -> e[1] == "ERROR", entries)
        println(stderr, "FAILED: $(basename(mdpath))" *
                        (isnothing(failure) ? "" : " ($(entries[failure][2]))"))
        return false
    end

    recordings = [e[2] for e in entries if e[1] == "RECORDING"]
    if !isempty(recordings) && isfile(cast)
        write_gifs(cast, recordings, figdir)
    end
    return true
end

"""
    check_references(mdpaths, figdir; complete)

Report figures that an example refers to but that were not generated, and (when
every example was run) figures in `figdir` that nothing refers to. Filenames are
the one part of the pipeline an author has to keep in step by hand, so they are
checked rather than left to be spotted in a built page.
"""
function check_references(mdpaths, figdir::AbstractString; complete::Bool)
    pattern = r"!\[[^\]]*\]\(assets/figures/([^)]+)\)"
    referenced = Dict{String, String}()
    for md in mdpaths, m in eachmatch(pattern, read(md, String))
        referenced[m.captures[1]] = basename(md)
    end
    present = Set(readdir(figdir))

    absent = sort([f for f in keys(referenced) if f ∉ present])
    if !isempty(absent)
        println(stderr, "\nReferenced but not generated:")
        foreach(f -> println(stderr, "  $f  (in $(referenced[f]))"), absent)
    end
    if complete
        unused = sort([f for f in present if !haskey(referenced, f)])
        if !isempty(unused)
            println("\nIn $(relpath(figdir)) but not referenced by any example:")
            foreach(f -> println("  $f"), unused)
        end
    end
    return isempty(absent)
end

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

function main(args)
    extract_only = "--extract-only" in args
    gif_only = "--gif-only" in args
    gif = !("--no-gif" in args)
    if "--no-train" in args
        # Nothing is trained, so there is nothing to record either.
        ENV["FIGGEN_NO_TRAIN"] = "1"
        gif = false
    end
    figdir = joinpath(DOCS, "src", "assets", "figures")
    if (i = findfirst(==("--figdir"), args)) !== nothing
        figdir = args[i + 1]
        deleteat!(args, [i, i + 1])
    end
    files = filter(a -> endswith(a, ".md"), args)
    isempty(files) && (files = sort(filter(f -> endswith(f, ".md"), readdir(EXAMPLES; join = true))))
    files = [isabspath(f) ? f : (isfile(f) ? abspath(f) : joinpath(EXAMPLES, basename(f))) for f in files]

    mkpath(figdir)
    failed = String[]
    for f in files
        ok = gif_only ? rebuild_gifs(f; figdir) : run_example(f; figdir, extract_only, gif)
        ok || push!(failed, basename(f))
    end

    println("\n", "="^78)
    if extract_only
        println("Extracted $(length(files)) examples to $(relpath(GENERATED))")
    elseif gif_only
        println("GIFs written to $(relpath(figdir))")
        isempty(failed) || println(stderr, "No recording for: $(join(failed, ", "))")
    else
        println("Figures written to $(relpath(figdir))")
        isempty(failed) ? println("All $(length(files)) examples succeeded") :
        println(stderr, "Failed: $(join(failed, ", "))")
        all_examples = length(files) == count(f -> endswith(f, ".md"), readdir(EXAMPLES))
        check_references(files, figdir; complete = isempty(failed) && all_examples)
    end
    return isempty(failed) ? 0 : 1
end

exit(main(copy(ARGS)))
