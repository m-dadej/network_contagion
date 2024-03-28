
bs = CSV.read("data/eba_stresstest2015.csv", DataFrame, header = 0, decimal = ',')
bs = sort(bs ./ 1_000_000, rev = true)

d = bs[:, 2]
e = bs[:, 1]#bs[1:5:50, 1]

n_sim = 20
target_params_ss = [0.04, 0.06, 0.12, 0.14] #0:0.05:0.8
target_params = [0.2]#0.2:0.05:0.4 #-collect(3.8:0.1:4.2)#-collect(0:0.1:6.0)

n_sim*length(target_params)*length(target_params_ss)

results = CSV.read("data/results_greed2.csv", DataFrame)

results = DataFrame(target        = Float64[],
                    target_ss     = Float64[],
                    n_default     = Int64[],
                    degree        = Float64[],
                    interm        = Int64[],
                    interm_assets = Float64[],
                    eq_r_l        = Float64[],
                    mean_liq      = Float64[],
                    mean_ib_share = Float64[],
                    mean_eq_req   = Float64[],
                    mean_n_share  = Float64[])

for target in target_params 
    for target_ss in target_params_ss
        for sim in 1:n_sim
            seed = rand(1:10000000000)
            Random.seed!(seed)

            println("seed: $seed | sim: $sim / $n_sim | trgt: $target / $(target_params[end]) | target_ss: $target_ss / $(target_params_ss[end])")
            
            bank_sys = BankSystem(α = 0.01,
                                    ω_n = 1.0, 
                                    ω_l = 0.8, 
                                    γ = 0.06,
                                    τ = 0.01, 
                                    ζ = 0.6, 
                                    exp_δ = 0.01, 
                                    σ_δ = 0.005)
            
            N = length(d)                  
            target_vec = super_greedy(target, target_ss, N)                  
            populate!(bank_sys, 
                        N = N, 
                        r_n = rand(Uniform(0.0, 0.15), N), 
                        σ = repeat([2.0], N), #rand(Normal(σ, 0.001), N), #rand([σ], length(d)),
                        d = d,
                        e = e,
                        target = e .* target_vec)   

            #super_spreader!(bank_sys, target_ss)
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
                fund_matching!(bank_sys, 0.2)    
            catch
                try
                    fund_matching!(bank_sys, 0.4)  
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
                
                #println("shocked bank $shocked_bank")
                bank_sys_scenario = deepcopy(bank_sys)

                res_sim = DataFrame(target = [target],
                                    target_ss = [target_ss],
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

balance_check(bank_sys)

plot_df = @chain results begin
    #transform(:σ_ss => x -> round.(x, digits=1), renamecols = false)
    groupby([:target_ss])
    #combine(:n_default => x -> sum(x .> 9)/sum(x .> 0))
    combine(:n_default => mean)
    sort()
    #unstack(:σ, :n_default_mean)
end

plot(plot_df.target_ss, plot_df.n_default_mean)

plot_df = @chain results begin
    #transform(:σ_ss => x -> round.(x, digits=1), renamecols = false)
    groupby([:target_ss])
    #combine(:n_default => x -> sum(x .> 9)/sum(x .> 0))
    combine(:mean_n_share => mean)
    sort()
    sort(:target_ss)
end

names(results)
plot(plot_df.target_ss, plot_df[:, 5])


plot_df = @chain results begin
    filter(:target => x -> x < 0.9, _)
    groupby([:target_ss, :target])
    #combine(:n_default => x -> sum(x .> 9)/sum(x .> 0))
    combine(:n_default => mean)
    #combine(nrow => :n_default_mean)
    filter(:target => x -> x == 0.2, _)
    filter(:target_ss => x -> x < 40, _)
    sort(:target_ss)
end

plot(plot_df.target_ss, plot_df.n_default_mean)

names(results)

CSV.write("data/results_greed2.csv", results)

results