


σ = 1.001
σ_ss = 0.0
Random.seed!(2)
d = [(606/1.06), (807/1.5), (529/1.08), (211/0.7), (838/1.47), (296/0.63), (250/0.68), (428/2), (284/1.24), (40/0.94), (8.2/0.2), (252/1.74), (24/0.19), (111.1/1.03), (88.9/1.3), (51.8/0.42), (63/0.48), (111.1/1.65), (100/1.37), (11.6/0.15)] # rand(Normal(700, 100), N) # deposits
e = [55.6, 90.0, 48.5, 53.0, 81.0, 53.0, 57.0, 48.0, 26.0, 43.0, 20.0, 23.0, 16.0, 10.0, 8.0, 5.0, 6.0, 10.0, 9.0, 9.0] #rand(Normal(50, 20), N) # equity

bank_sys = BankSystem(α = 0.05,
                        ω_n = 1.2, 
                        ω_l = 0.5, 
                        γ = 0.06,
                        τ = 0.025, 
                        ζ = 0.5, 
                        exp_δ = 0.01, 
                        σ_δ = 0.01)

populate!(bank_sys, 
            N = length(d), 
            r_n = rand(Uniform(0.0, 0.15), length(d)), 
            σ = rand([σ], length(d)),
            d = d,
            e = e)   

super_spreader!(bank_sys, σ_ss)

equilibrium!(bank_sys)    
maximum(balance_check(bank_sys, "book"))   
#maximum(balance_check(bank_sys, "book")) > 0.001 && @warn "balance sheet identity not satisfied"; continue
println("max BS diff: ", maximum(balance_check(bank_sys, "book")))
get_market_balance(bank_sys)
adjust_imbalance!(bank_sys)

fund_matching!(bank_sys, 0.3)  

sum(bank_sys.A_ib[i,:]'equity_requirement(bank_sys) for i in 1:length(bank_sys.banks))



[bank.σ = 2.001 for bank in bank_sys.banks]
A_ib = rand(1:1:100, 10,10)
k = collect(0.1:0.1:1)

1.0 * A_ib[1,:]'k
