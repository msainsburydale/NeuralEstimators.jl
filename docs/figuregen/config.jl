# Configuration for the example figure pipeline.
#
# Every rule that governs regeneration lives here or in extract.jl; the example
# markdown files in docs/src/examples contain no directives of any kind.

# Examples that take a long time to run, and that run.sh therefore asks about
# before including them. Everything else runs at the settings written in the docs.
const EXPENSIVE = [
    "data_gridded_nonstationary.md",
    "data_missing_censored.md",
    "data_spatiotemporal.md"
]

# Short suffixes for the section headings that recur across the examples, so that
# figures are named `gridded_data.png` rather than `gridded_simulating-data.png`.
# Headings that are not listed fall back to their slugified form.
const HEADING_ALIASES = Dict(
    "Simulating data" => "data",
    "Sampling parameters and simulating data" => "data",
    "Bonus: Visualizing spatio-temporal dependence" => "data",
    "Training the estimator" => "training",
    "Assessing the estimator" => "assessment",
    "Applying the estimator to observed data" => "application"
)

# Escape hatch for blocks that cannot be run. A block is dropped if its code
# matches any pattern listed against its markdown file.
const BLOCK_FILTERS = Dict{String, Vector{Regex}}()

# Recording settings. The width is chosen so that train()'s per-epoch status line
# fits without being truncated, which it does at around 155 characters. The
# recorded terminal is taller than the training display needs, so the GIF is
# rendered at GIF_ROWS instead; that display never fills more than nine rows.
const TERM_COLS = 160      # width of the recorded terminal
const TERM_ROWS = 14       # height of the recorded terminal
const GIF_ROWS = 9         # height the GIF is rendered at
const GIF_FONT_SIZE = 16
const GIF_FPS_CAP = 10     # frames per second; bounds the size of a long GIF
const IDLE_TIME_LIMIT = 0.5   # seconds; longer pauses are compressed to this

# Training runs can last many minutes. Rather than cut anything out, a recording
# longer than this is played faster, so that every epoch is still shown.
const GIF_TARGET_SECONDS = 25
