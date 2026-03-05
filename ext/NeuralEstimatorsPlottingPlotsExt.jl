module NeuralEstimatorsPlottingPlotsExt

using NeuralEstimators
using Plots


# ===========================================================================
#  plotrisk()
# ===========================================================================
import NeuralEstimators: _plotrisk
function _plotrisk(savepath::String)
    history = loadrisk(savepath)
    epochs = 0:(size(history, 1) - 1)
    p = plot(epochs, history[:, 1], label = "Training", xlabel = "Epoch", ylabel = "Empirical risk (average loss)", linewidth = 2)
    plot!(p, epochs, history[:, 2], label = "Validation", linewidth = 2)
    return p
end

# ===========================================================================
#  plot(assessment)
# ===========================================================================



end  # module
