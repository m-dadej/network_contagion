using Pkg
#Pkg.add.(["DataFramesMeta", "Plots", "DelimitedFiles", "CSV", "DataFrames"])
using DelimitedFiles
using Plots
using CSV
using DataFrames
using DataFramesMeta 
using Plots

# cioe
# jak uzywac sbagliato


include("risk_heterogeneity.jl")
include("optim_alloc_nlopt.jl")
#include("optim_alloc_jump.jl")


d = [(606/1.06), (807/1.5), (529/1.08), (211/0.7), (838/1.47), (296/0.63), (250/0.68), (428/2), (284/1.24), (40/0.94), (8.2/0.2), (252/1.74), (24/0.19), (111.1/1.03), (88.9/1.3), (51.8/0.42), (63/0.48), (111.1/1.65), (100/1.37), (11.6/0.15)] # rand(Normal(700, 100), N) # deposits
e = [55.6, 90.0, 48.5, 53.0, 81.0, 53.0, 57.0, 48.0, 26.0, 43.0, 20.0, 23.0, 16.0, 10.0, 8.0, 5.0, 6.0, 10.0, 9.0, 9.0] #rand(Normal(50, 20), N) # equity

bs = CSV.read("data/eba_stresstest2015.csv", DataFrame, header = 0, decimal = ',')
bs = sort(bs ./ 1_000_000, rev = true)

d = bs[:, 2]
e = bs[:, 1]#bs[1:5:50, 1]

n_sim = 50
σ_ss_params = -collect(3.8:0.1:4.2)#-collect(0:0.1:6.0)
σ_params = [4.0] .+ 0.001

n_sim*length(σ_ss_params)*length(σ_params)

results = CSV.read("data/results_done.csv", DataFrame)

results = DataFrame(σ             = Float64[],
                    σ_ss          = Float64[],
                    n_default     = Int64[],
                    degree        = Float64[],
                    interm        = Int64[],
                    interm_assets = Float64[],
                    eq_r_l        = Float64[],
                    mean_liq      = Float64[],
                    mean_ib_share = Float64[],
                    mean_eq_req   = Float64[],
                    mean_n_share  = Float64[])

for σ_ss in σ_ss_params
    for σ in σ_params
        for sim in 1:n_sim
            seed = rand(1:10000000000)
            #Random.seed!(seed)
            println("seed: $seed | sim: $sim / $n_sim | σ: $σ / $(σ_params[end]) | σ_ss: $σ_ss / $(σ_ss_params[end])")
            
            bank_sys = BankSystem(α = 0.01,
                                    ω_n = 1.0, 
                                    ω_l = 0.5, 
                                    γ = 0.06,
                                    τ = 0.02, 
                                    ζ = 0.6, 
                                    exp_δ = 0.005, 
                                    σ_δ = 0.003)
            
            N = length(d)                                    
            populate!(bank_sys, 
                        N = N, 
                        r_n = rand(Uniform(0.0, 0.15), N), 
                        σ = rand(Normal(σ, 0.1), N), #rand([σ], length(d)),
                        d = d,
                        e = e)   

            super_spreader!(bank_sys, σ_ss)
            try 
                equilibrium!(bank_sys, verbose = false, min_iter = 50)      
            catch    
                @warn "equalibrium error"
                continue
            end    

            if maximum(balance_check(bank_sys)) > 0.001
                @warn "balance sheet identity not satisfied"
                continue
            end                

            if abs(get_market_balance(bank_sys)) > N*50
                @warn "market imbalance too high: $(round(get_market_balance(bank_sys))) -> no equilibrium convergence"
                continue
            end

            println("max BS diff: ", maximum(balance_check(bank_sys)), " | imbalance: $(round(get_market_balance(bank_sys)))")
            adjust_imbalance!(bank_sys)
            try
                fund_matching!(bank_sys, 0.1)    
            catch
                try
                    fund_matching!(bank_sys, 0.3)  
                catch
                    @warn "NO fund_matching solution!"
                    continue
                end
            end

            if length(intermediators(bank_sys)) != 0
                interm_ass = min(bank_sys.banks[intermediators(bank_sys)[1]].b,
                             bank_sys.banks[intermediators(bank_sys)[1]].l)
            else
                interm_ass = 0
            end                

            for shocked_bank in 1:N
                
                println("shocked bank $shocked_bank")
                bank_sys_scenario = deepcopy(bank_sys)

                res_sim = DataFrame(σ = [σ],
                                    σ_ss = [σ_ss],
                                    n_default = [0],
                                    degree = [degree(bank_sys)],
                                    interm = [length(intermediators(bank_sys))],
                                    interm_assets = [interm_ass],
                                    eq_r_l = [bank_sys.r_l],
                                    mean_liq = [mean(liquidity(bank_sys))],
                                    mean_ib_share = [mean(ib_share(bank_sys))],
                                    mean_eq_req = [mean(equity_requirement(bank_sys))],
                                    mean_n_share = [mean(bank.n / assets(bank, bank_sys) for bank in bank_sys.banks)])
                        
                contagion_liq!(bank_sys_scenario, shocked_bank)

                res_sim.n_default[1] = n_default(bank_sys_scenario)
                results = [results; res_sim]
            end                            
        end
    end
end


plot_df = @chain results begin
    transform(:σ_ss => x -> round.(x, digits=1), renamecols = false)
    groupby([:σ_ss])
    #combine(:n_default => x -> sum(x .> 9)/sum(x .> 0))
    combine(:n_default => mean)
    sort()
    #unstack(:σ, :n_default_mean)
end

plot_df[1:30,:]

plot(plot_df[:,1], plot_df[:,2])

OneSampleTTest(results[results.σ_ss .!= 0.0,:].n_default[5000:10200], 
               results[results.σ_ss .== 0.0,:].n_default[5000:10200])


heatmap(σ_ss_params, σ_params, Matrix(heatmap_df)[:,2:end])

heatmap_df = @chain results begin
    transform(:σ_ss => x -> round.(x), renamecols = false)
    groupby([:σ, :σ_ss])
    combine(:n_default => mean)
    sort()
    #unstack(:σ, :n_default_mean)
end

results.eq_r_l

CSV.write("data/results_done.csv", results)

quantile(results.n_default, [0.5, 0.75, 0.8, 0.9, 0.95, 0.99, 1.0])

heatmap(σ_ss_params, σ_params, Matrix(heatmap_df)[:,2:end])

@chain results[40000:end,:] begin
    groupby([:σ, :σ_ss])
    combine(:n_default => mean)
    sort()
    unstack(:σ, :n_default_mean)
    sort()
end    

@chain results begin
    groupby([:σ_ss])
    #combine([:n_default, :interm, :degree, :eq_r_l] .=> mean)
    combine([:mean_n_share, :mean_liq, :mean_ib_share] .=> mean)
    sort()
end    

results.interm_assets

cor(results.n_default, results.degree)

names(select(results, Not(:σ_ss)))
names(results)
@chain results begin
    #select(Not(:σ_ss))
    select([:n_default, :σ_ss])
    Matrix()
    cor()
end


@chain results begin
    groupby([:σ_ss])
    #combine(:n_default => x -> sum(x .> 10)/sum(x .> 0))
    combine(nrow => :count)
    sort()
    #unstack(:σ, :n_default_function)
end    

 sort(combine(groupby(results, [:σ, :σ_ss]), [:n_default, :degree] .=> mean))
 sort(combine(groupby(results, [:σ, :σ_ss]), [:n_default] .=> x -> sum(x .> 2)/sum(x .> 0)))



mean([1,2,3,4]) |> round()

sort(combine(groupby(results, [:σ, :σ_ss]), [:n_default, :degree, :eq_r_l, :mean_liq] .=> mean))

combine(groupby(results, [:σ_ss, :σ]), :n_default => x -> sum(x .> 2)/sum(x .> 0))
combine(groupby(results, :σ), :n_default => x -> sum(x .> 2)/sum(x .> 0))

CSV.write("results_nlopt.csv", results)



combine(groupby(results, :σ_ss), [:interm, :mean_ib_share] .=> mean)
combine(groupby(results, [:σ_ss, :σ]), nrow => :count)


get_market_balance(bank_sys)

intermediators(bank_sys)




liquidity(bank_sys)
equity_requirement(bank_sys)
assets(bank_sys)
ib_share(bank_sys)
leverage(bank_sys)


[bank_sys.banks[i].e for i in 1:N]

n_default(bank_sys)
get_market_balance(bank_sys)
bank_sys.banks
bank_sys.A_ib

