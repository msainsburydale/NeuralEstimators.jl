#!/usr/bin/env bash
#
# Regenerate the figures, training recordings and diagnostic plots for the
# examples in docs/src/examples. The Julia code is extracted from the markdown
# files themselves, so this script never needs to be kept in step with them.
#
#   docs/figuregen/run.sh                  # prompt about the expensive examples
#   docs/figuregen/run.sh --all            # include them without asking
#   docs/figuregen/run.sh --skip-expensive # exclude them without asking
#   docs/figuregen/run.sh --only data_replicated.md
#   docs/figuregen/run.sh --extract-only   # write the extracted scripts and stop
#   docs/figuregen/run.sh --no-train       # stop before training: data figures only
#   docs/figuregen/run.sh --no-gif         # skip the terminal recordings
#   docs/figuregen/run.sh --gif-only       # redo the GIFs from the last run's recordings

set -euo pipefail

FIGUREGEN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES="$(dirname "$FIGUREGEN")/src/examples"
export PATH="$HOME/.local/bin:$PATH"

INCLUDE_EXPENSIVE=ask
ONLY=()
PASS_THROUGH=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all) INCLUDE_EXPENSIVE=yes; shift ;;
        --skip-expensive) INCLUDE_EXPENSIVE=no; shift ;;
        --only) ONLY+=("$(basename "$2")"); shift 2 ;;
        --extract-only|--no-train|--no-gif|--gif-only) PASS_THROUGH+=("$1"); shift ;;
        --figdir) PASS_THROUGH+=("$1" "$2"); shift 2 ;;
        # Print the comment block above, which is the usage message.
        -h|--help) awk 'NR > 1 && !/^#/ {exit} NR > 1 {sub(/^# ?/, ""); print}' \
            "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------

command -v julia >/dev/null || { echo "julia is not on PATH" >&2; exit 1; }

GIF_ONLY=no
[[ " ${PASS_THROUGH[*]-} " == *" --gif-only "* ]] && GIF_ONLY=yes

# --gif-only reuses the recordings of an earlier run, so nothing is computed and
# every example is cheap regardless of how expensive it is to run.
[[ "$GIF_ONLY" == yes ]] && INCLUDE_EXPENSIVE=yes

if [[ "$GIF_ONLY" == no ]] && ! [[ " ${PASS_THROUGH[*]-} " == *" --extract-only "* ]]; then
    if ! nvidia-smi >/dev/null 2>&1; then
        echo "Warning: no working NVIDIA GPU was found. The examples load CUDA (the"
        echo "first entry of each GPU code-group), so they will run on the CPU at best"
        echo "and may fail outright. Run this on a GPU machine for usable timings."
        echo
    fi

    if ! [[ " ${PASS_THROUGH[*]-} " == *" --no-gif "* ]]; then
        missing=()
        command -v asciinema >/dev/null || missing+=(asciinema)
        command -v agg >/dev/null || missing+=(agg)
        if [[ ${#missing[@]} -gt 0 ]]; then
            echo "Missing recording tools: ${missing[*]}"
            echo "  asciinema:  pip install --user asciinema      (or brew install asciinema)"
            echo "  agg:        curl -fsSL -o ~/.local/bin/agg \\"
            echo "                https://github.com/asciinema/agg/releases/download/v1.5.0/agg-x86_64-unknown-linux-gnu \\"
            echo "              && chmod +x ~/.local/bin/agg      (or brew install agg)"
            echo
            echo "Without asciinema the examples still run but no recording is made;"
            echo "without agg the recording is rendered as a still PNG instead of a GIF."
            read -r -p "Continue anyway? [y/N] " reply
            [[ "$reply" =~ ^[Yy]$ ]] || exit 1
        fi
    fi
elif [[ "$GIF_ONLY" == yes ]]; then
    command -v agg >/dev/null || { echo "agg is not on PATH: nothing to render with" >&2; exit 1; }
fi

# ---------------------------------------------------------------------------
# Which examples to run
# ---------------------------------------------------------------------------

# The list of expensive examples lives in config.jl, so it is read from there
# rather than duplicated here.
mapfile -t EXPENSIVE < <(julia --startup-file=no -e \
    "include(\"$FIGUREGEN/config.jl\"); foreach(println, EXPENSIVE)")

if [[ ${#ONLY[@]} -gt 0 ]]; then
    FILES=("${ONLY[@]}")
else
    mapfile -t FILES < <(cd "$EXAMPLES" && ls *.md)

    expensive_selected=()
    for f in "${FILES[@]}"; do
        for e in "${EXPENSIVE[@]}"; do
            [[ "$f" == "$e" ]] && expensive_selected+=("$f")
        done
    done

    if [[ ${#expensive_selected[@]} -gt 0 && "$INCLUDE_EXPENSIVE" == ask ]]; then
        echo "These examples are computationally expensive:"
        printf '  %s\n' "${expensive_selected[@]}"
        if [[ -t 0 ]]; then
            read -r -p "Include them? [y/N] " reply
            [[ "$reply" =~ ^[Yy]$ ]] && INCLUDE_EXPENSIVE=yes || INCLUDE_EXPENSIVE=no
        else
            INCLUDE_EXPENSIVE=no
        fi
        echo
    fi

    if [[ "$INCLUDE_EXPENSIVE" == no ]]; then
        keep=()
        for f in "${FILES[@]}"; do
            skip=no
            for e in "${EXPENSIVE[@]}"; do
                [[ "$f" == "$e" ]] && skip=yes
            done
            [[ "$skip" == no ]] && keep+=("$f")
        done
        FILES=("${keep[@]}")
    fi
fi

echo "Running: ${FILES[*]}"
exec julia --startup-file=no --project="$FIGUREGEN" "$FIGUREGEN/generate.jl" \
    ${PASS_THROUGH[@]+"${PASS_THROUGH[@]}"} "${FILES[@]}"
