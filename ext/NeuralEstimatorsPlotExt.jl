module NeuralEstimatorsPlotExt 

using NeuralEstimators 
using AlgebraOfGraphics
using CairoMakie
import CairoMakie: plot
export plot # method for Assessment objects
using ColorSchemes

"""
    plot(assessment::Assessment; grid::Bool = false) 

Method for visualising the performance of a neural estimator (or multiple neural estimators). 

One may set `grid=true` to facet the figure based on the estimator. 
"""
function plot(assessment::Assessment; grid::Bool = false) 

	df = assessment.df
	num_estimators = "estimator" ∉ names(df) ? 1 : length(unique(df.estimator))
  
	# figure needs to be created first so that we can add to it below
	# NB code rep, we have the same call towards the end of the function... is there a better way to initialise an empty figure?
	figure = mapping([0], [1]) * visual(ABLines, color=:red, linestyle=:dash)

	# Code for QuantileEstimators 
	#TODO multiple estimators (need to incorporate code below)
	if "prob" ∈ names(df)
		df = empiricalprob(assessment)
		figure = mapping([0], [1]) * visual(ABLines, color=:red, linestyle=:dash) 
		figure += data(df) * mapping(:prob, :empirical_prob, layout = :parameter) * visual(Lines, color = :black)
		figure = draw(figure, facet=(; linkxaxes=:none, linkyaxes=:none), axis = (; xlabel="Probability level, τ", ylabel="Pr(Q(Z, τ) ≤ θ)"))
		return figure 
	end
  
	if all(["lower", "upper"] .∈ Ref(names(df)))
		# Need line from (truth, lower) to (truth, upper). To do this, we need to
		# merge lower and upper into a single column and then group by k.
		df = stack(df, [:lower, :upper], variable_name = :bound, value_name = :interval)
		figure += data(df) * mapping(:truth, :interval, group = :k => nonnumeric, layout = :parameter) * visual(Lines, color = :black)
		figure += data(df) * mapping(:truth, :interval, layout = :parameter) * visual(Scatter, color = :black, marker = '⎯')
	end
  
	linkyaxes=:none
	if "estimate" ∈ names(df) #TODO only want this for point estimates 
		if num_estimators > 1
		  colors = [unique(df.estimator)[i] => ColorSchemes.Set1_4.colors[i] for i ∈ 1:num_estimators]
		  if grid
			figure += data(df) * mapping(:truth, :estimate, color = :estimator, col = :estimator, row = :parameter) * visual(palettes=(color=colors,), alpha = 0.75)
			linkyaxes=:minimal
		  else 
			figure += data(df) * mapping(:truth, :estimate, color = :estimator, layout = :parameter) * visual(palettes=(color=colors,), alpha = 0.75)
			linkyaxes=:none
		  end
		else
		  figure += data(df) * mapping(:truth, :estimate, layout = :parameter) * visual(color = :black, alpha = 0.75)
		end
	end
  
	figure += mapping([0], [1]) * visual(ABLines, color=:red, linestyle=:dash) 
	figure = draw(figure, facet=(; linkxaxes=:none, linkyaxes=linkyaxes)) #, axis=(; aspect=1)) # couldn't fix the aspect ratio without messing up the positioning of the titles
	return figure
end

# using CairoMakie # for save()
# figure = plot(assessment)
# save("docs/src/assets/figures/gridded.png", figure, px_per_unit = 3, size = (600, 300))
# save("GNN.png", figure, px_per_unit = 3, size = (450, 450))

end