#NB deprecated because it isn't the recommended way of storing models anymore
"""
	loadbestweights(path::String)

Returns the weights of the neural network saved as 'best_network.bson' in the given `path`.
"""
loadbestweights(path::String) = loadweights(joinpath(path, "best_network.bson"))
loadweights(path::String) = load(path, @__MODULE__)[:weights]


# aliases for backwards compatability
WeightedGraphConv = SpatialGraphConv; export WeightedGraphConv 
simulategaussianprocess = simulategaussian; export simulategaussianprocess