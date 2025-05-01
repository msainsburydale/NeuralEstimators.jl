####################################################################################################
#                                      load packages
####################################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("")
using BenchmarkPlots
using DataFrames
using PkgBenchmark
using NeuralEstimators
####################################################################################################
#                                       benchmark
####################################################################################################
baselin_id = "1a4c67fb8a82076cecd25d0ba75072f3218ed354"
baseline = benchmarkpkg(NeuralEstimators, baselin_id)

target_id = "d784a53c61872319a6c6c7298fd80e3571335fde"
target = benchmarkpkg(NeuralEstimators, target_id)

comparison = judge(target, baseline)
export_markdown("judgement_test.md", comparison)
