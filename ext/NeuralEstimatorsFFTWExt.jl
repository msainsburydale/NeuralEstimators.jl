module NeuralEstimatorsFFTWExt

using NeuralEstimators
using FFTW
import NeuralEstimators: variogram

variogram(Z::AbstractMatrix; n_bins::Int = 20, maxlag = 0.5) = vec(variogram(reshape(Z, size(Z, 1), size(Z, 2), 1); n_bins, maxlag))

function variogram(Z::AbstractArray{<:Real, 3}; n_bins::Int = 20, maxlag = 0.5)
    (0 < maxlag <= 1) || throw(ArgumentError("maxlag must be in (0, 1]"))
    nx, ny, K = size(Z)
    T = float(eltype(Z))
    px = nextpow(2, 2nx - 1)
    py = nextpow(2, 2ny - 1)

    P = zeros(T, px, py, K)
    @inbounds P[1:nx, 1:ny, :] .= Z
    P2 = P .^ 2

    M = zeros(T, px, py, 1)
    @inbounds M[1:nx, 1:ny, 1] .= one(T)

    Fp = FFTW.rfft(P, 1:2)
    F2 = FFTW.rfft(P2, 1:2)
    FM = FFTW.rfft(M, 1:2)

    C = FFTW.irfft(Fp .* conj.(Fp), px, 1:2)
    Q1 = FFTW.irfft(F2 .* conj.(FM), px, 1:2)
    Q2 = FFTW.irfft(FM .* conj.(F2), px, 1:2)
    S = reshape(Q1 .+ Q2 .- 2 .* C, px * py, K)

    idxs, bins, counts = _variogram_lagmap(nx, ny, px, py, n_bins, maxlag)
    sums = zeros(T, n_bins, K)
    @inbounds for t in eachindex(idxs)
        b = bins[t]
        i = idxs[t]
        for k = 1:K
            sums[b, k] += S[i, k]
        end
    end
    sums ./ (2 .* counts)
end

# Unique unordered lags on the FFT grid, with isotropic distance bins and
# closed-form pair counts for a complete rectangular lattice.
function _variogram_lagmap(nx::Int, ny::Int, px::Int, py::Int, n_bins::Int, maxlag)
    hmax = maxlag * hypot(nx - 1, ny - 1)
    edges = range(0, hmax; length = n_bins + 1)
    idxs = Int[]
    bins = Int[]
    counts = zeros(Int, n_bins)
    sizehint!(idxs, nx * ny)
    sizehint!(bins, nx * ny)
    @inbounds for j = 1:py
        dy = j - 1
        dy > py ÷ 2 && (dy -= py)
        absdy = abs(dy)
        absdy >= ny && continue
        for i = 1:px
            dx = i - 1
            dx > px ÷ 2 && (dx -= px)
            absdx = abs(dx)
            absdx >= nx && continue
            (dy > 0 || (dy == 0 && dx > 0)) || continue
            n = (nx - absdx) * (ny - absdy)
            n == 0 && continue
            h = hypot(dx, dy)
            h > hmax && continue
            b = min(searchsortedlast(edges, h), n_bins)
            push!(idxs, i + px * (j - 1))
            push!(bins, b)
            counts[b] += n
        end
    end
    idxs, bins, counts
end

end
