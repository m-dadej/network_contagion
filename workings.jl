using Pkg
#Pkg.add.(["DelimitedFiles", "CSV", "DataFrames"])
using DelimitedFiles
using CSV
using DataFrames

include("risk_heterogeneity.jl")
include("optim_alloc_nlopt.jl")
#include("optim_alloc_jump.jl")


seed = rand(1:1000)

Random.seed!(743)
println(seed)
bank_sys = BankSystem(α = 0.05,
                        ω_n = 1.2, 
                        ω_l = 0.5, 
                        γ = 0.06,
                        τ = 0.025, 
                        ζ = 0.6, 
                        exp_δ = 0.005, 
                        σ_δ = 0.003)
            
populate!(bank_sys, 
            N = length(d), 
            r_n = rand(Uniform(0.0, 0.15), length(d)), 
            σ = rand([σ], length(d)),
            d = d,
            e = e)   

equilibrium!(bank_sys, verbose = false)                     
println("max BS diff: ", maximum(balance_check(bank_sys, "book")))
get_market_balance(bank_sys)
adjust_imbalance!(bank_sys)

fund_matching!(bank_sys, 0.2)    

contagion_liq!(bank_sys)

sum([bank.e <= 0.00001 for bank in bank_sys.banks])

