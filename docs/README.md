# Instructions for contributing to the documentation

The [package documentation](https://msainsburydale.github.io/NeuralEstimators.jl/dev/) is built using [Documenter.jl](https://documenter.juliadocs.org/stable/). Source files for the documentation are located in [docs/src](https://github.com/msainsburydale/NeuralEstimators.jl/tree/main/docs/src) and as the docstrings for each function or type defined in [src](https://github.com/msainsburydale/NeuralEstimators.jl/tree/main/src). 

### Workflow for contributing

1.	**Download the source code**: Clone the package repository from GitHub, for instance, by running the following command:
```bash
git clone https://github.com/msainsburydale/NeuralEstimators.jl.git
```
2.	**Edit the documentation**: Modify the relevant source files (in `docs/src`) or docstrings (in `src`).
3.	**(Optional) Build the documentation locally**: Preview your changes by building the documentation locally. This can be done by: 
 - navigating to `docs/` (i.e., running `cd docs`)
 - Installing the relevant packages if not already on your system: 
 ```bash
julia -e 'using Pkg; Pkg.add(["Documenter", "DocumenterVitepress", "LiveServer"])'
```
 - Then running the following command:
```bash
julia --project=. make.jl && julia -e 'using LiveServer; serve(dir="build/1")'
```
4.	**(Optional) Regenerate the example figures**: If you changed the code in an example, regenerate its figures as described in [Regenerating example figures](#regenerating-example-figures) below.
5.	**Push changes**: Once satisfied with your changes, `git commit` and `git push` to the main branch. The updated documentation will be automatically built and deployed.

## Regenerating example figures

Each example in `docs/src/examples` shows figures of the simulated data, a recording of the terminal output during training, the empirical risk over the training run, and diagnostic plots of the trained estimator. These are all produced by [docs/figuregen/run.sh](figuregen/run.sh):

```bash
docs/figuregen/run.sh                            # all examples, prompting about the expensive ones
docs/figuregen/run.sh --only data_replicated.md   # a single example
docs/figuregen/run.sh --no-train                 # stop before training: data figures only
docs/figuregen/run.sh --extract-only             # write the extracted scripts and stop
docs/figuregen/run.sh --gif-only                 # redo the GIFs from the last run's recordings
```

`--gif-only` is worth knowing about: the full terminal recording of each example is kept in `docs/figuregen/generated`, so the appearance of the GIFs can be changed and rebuilt in about a minute without retraining anything.

Figures are written to `docs/src/assets/figures`, which `make.jl` copies into `docs/src/examples/assets` at build time; that is why the examples refer to their figures as `assets/figures/NAME`.

### Prerequisites

- A machine with a working NVIDIA GPU. The first entry of each GPU code-group is `using CUDA, cuDNN`, and it is that entry the pipeline runs (see below), so the examples train on the GPU.
- [asciinema](https://asciinema.org) to record the terminal, and [agg](https://github.com/asciinema/agg) to turn the recording into a GIF:

```bash
pip install --user asciinema     # or brew install asciinema
curl -fsSL -o ~/.local/bin/agg \
  https://github.com/asciinema/agg/releases/download/v1.5.0/agg-x86_64-unknown-linux-gnu \
  && chmod +x ~/.local/bin/agg   # or brew install agg
```

Without `asciinema` the examples still run and their figures are still written, but no recording is made. Without `agg` the recording is rendered as a still PNG instead of a GIF.

The Julia packages the examples need are declared in `docs/figuregen/Project.toml`, separately from the documentation build itself.

### What the pipeline expects of an example

The code is extracted from the markdown rather than kept in a separate script, so the pipeline has to infer what to do from the page. Four conventions matter when editing an example:

1. **Only the first entry of a `::: code-group` is run.** That entry therefore has to be the self-contained one. This is also how the GPU backend is chosen.
2. **A figure is saved only if the block produces one**, by ending in the figure (`fig`), displaying it (`display(fig)`), or returning one from a call such as `plot(assessment)`. A figure left in a global variable is also picked up.
3. **Recording is triggered by a call to `train()`**, and a figure from a block that calls `plotrisk()` is named as the risk curve. Nothing else is treated specially.
4. **Figure filenames are derived from the file and the section heading**: `data_replicated.md` plus `## Assessing the estimator` gives `replicated_assessment.png`. Renaming a heading therefore renames the figure, and the image reference has to be updated to match. Short names for the recurring headings come from `HEADING_ALIASES` in `docs/figuregen/config.jl`; a second figure under one heading gets a `_2` suffix.

Every run prints the name of each figure as it is written, and finishes by listing any figure an example refers to but that was not generated, so a mistyped or stale reference is caught here rather than in a built page.

### If something goes wrong

The extracted code for each example is written to `docs/figuregen/generated/NAME.jl`. It is an ordinary script, with `#= FIGGEN BLOCK n =#` separators showing which markdown block each part came from, so a failing example can be inspected and rerun by hand. When a block raises an error, the pipeline prints the block and the markdown line it starts at, and moves on to the next example.

`docs/figuregen/config.jl` holds the settings: which examples are treated as expensive, the size of the recorded terminal, and a `BLOCK_FILTERS` table for excluding a block that cannot run. It also holds the two settings that govern how a recording becomes a GIF. `GIF_TARGET_SECONDS` is the longest a GIF may play for; a training run that took longer than that is played faster rather than having part of it cut out, so every epoch is still shown. `GIF_ROWS` is the height the GIF is rendered at, which is smaller than the height of the recorded terminal because the training display never fills more than nine rows.

