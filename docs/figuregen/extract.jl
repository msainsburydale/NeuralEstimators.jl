# Extraction of runnable Julia code from the example markdown files.
#
# The markdown is the single source of truth: this file contains all of the rules
# needed to turn it into a script, so that no tooling syntax has to be added to
# the documentation itself.
#
# The rules are:
#   - every ```julia block is run, in document order;
#   - inside a `::: code-group`, only the first ```julia block is run;
#   - a block that calls train() is recorded, to be turned into a GIF;
#   - figures are named after the file and the nearest preceding heading.

struct Block
    code::String
    heading::String
    line::Int         # line of the opening fence, for error messages
    records::Bool     # calls train(), so the terminal output is worth recording
    plots_risk::Bool  # calls plotrisk(), so its figure is named accordingly
end

const _JULIA_FENCE = r"^```julia(\s*\[[^\]]*\])?\s*$"
const _HEADING = r"^(#{1,6})\s+(.*?)\s*$"

"""
    extract_blocks(mdpath)

Return the [`Block`](@ref)s of runnable Julia code in the markdown file `mdpath`.
"""
function extract_blocks(mdpath::AbstractString)
    lines = readlines(mdpath)
    filters = get(BLOCK_FILTERS, basename(mdpath), Regex[])

    blocks = Block[]
    heading = ""
    in_group = false
    group_taken = false

    i = 1
    while i <= length(lines)
        line = lines[i]

        # Fenced blocks are consumed whole, so headings and directives are never
        # confused with Julia comments.
        if startswith(line, "```")
            is_julia = match(_JULIA_FENCE, line) !== nothing
            j = i + 1
            while j <= length(lines) && !startswith(lines[j], "```")
                j += 1
            end
            code = join(lines[(i + 1):(j - 1)], "\n")
            if is_julia && !(in_group && group_taken) && !any(p -> occursin(p, code), filters)
                push!(blocks, Block(code, heading, i, calls(code, r"\btrain\("),
                    calls(code, r"\bplotrisk\(")))
                in_group && (group_taken = true)
            end
            i = j + 1
            continue
        end

        m = match(_HEADING, line)
        if m !== nothing
            # The level-one heading is the page title, not a section.
            heading = length(m.captures[1]) == 1 ? "" : String(m.captures[2])
            i += 1
            continue
        end

        stripped = strip(line)
        if stripped == "::: code-group"
            in_group = true
            group_taken = false
        elseif stripped == ":::"
            in_group = false
        end
        i += 1
    end

    return blocks
end

"""
    calls(code, pattern)

Whether `code` matches `pattern` outside of a comment. Used to spot the calls that
the pipeline keys off, namely `train()` and `plotrisk()`.
"""
function calls(code::AbstractString, pattern::Regex)
    for line in eachsplit(code, '\n')
        uncommented = first(eachsplit(line, '#'))
        occursin(pattern, uncommented) && return true
    end
    return false
end

"""
    fileslug(mdpath)

Prefix used for every figure generated from `mdpath`, e.g. `data_replicated.md`
gives `replicated`.
"""
fileslug(mdpath::AbstractString) = replace(splitext(basename(mdpath))[1], r"^data_" => "")

function slugify(s::AbstractString)
    s = replace(s, r"\$[^$]*\$" => "", "`" => "")
    s = replace(lowercase(s), r"[^a-z0-9]+" => "-")
    return String(strip(s, '-'))
end

"""
    headingslug(heading)

Figure-name component for a section heading: the alias from `HEADING_ALIASES` if
there is one, and otherwise the slugified heading.
"""
headingslug(heading::AbstractString) = get(HEADING_ALIASES, heading, slugify(heading))

"""
    script_text(mdpath, blocks)

The extracted code as a standalone script. Written to `generated/` purely so that
a failing example can be inspected and rerun by hand; the pipeline itself runs
the blocks one at a time.
"""
function script_text(mdpath::AbstractString, blocks::Vector{Block})
    io = IOBuffer()
    println(io, "# Extracted from docs/src/examples/$(basename(mdpath)) by docs/figuregen.")
    println(io, "# Generated file: edit the markdown, not this script.")
    for (n, b) in enumerate(blocks)
        section = isempty(b.heading) ? "(page preamble)" : b.heading
        println(io)
        println(io, "#= FIGGEN BLOCK $n: $section (line $(b.line))$(b.records ? " [recorded]" : "") =#")
        println(io, b.code)
    end
    return String(take!(io))
end
