using Pkg
#Pkg.add.(["DataFramesMeta", "Plots", "DelimitedFiles", "CSV", "DataFrames"])
using DelimitedFiles
using Plots
using CSV
using DataFrames
using DataFramesMeta 

include("risk_heterogeneity.jl")
include("optim_alloc_nlopt.jl")
#include("optim_alloc_jump.jl")

d = [(606/1.06), (807/1.5), (529/1.08), (211/0.7), (838/1.47), (296/0.63), (250/0.68), (428/2), (284/1.24), (40/0.94), (8.2/0.2), (252/1.74), (24/0.19), (111.1/1.03), (88.9/1.3), (51.8/0.42), (63/0.48), (111.1/1.65), (100/1.37), (11.6/0.15)] # rand(Normal(700, 100), N) # deposits
e = [55.6, 90.0, 48.5, 53.0, 81.0, 53.0, 57.0, 48.0, 26.0, 43.0, 20.0, 23.0, 16.0, 10.0, 8.0, 5.0, 6.0, 10.0, 9.0, 9.0] #rand(Normal(50, 20), N) # equity

bs = CSV.read("data/eba_stresstest2015.csv", DataFrame, header = 0, decimal = ',')
bs = sort(bs ./ 1_000_000, rev = true)

d = bs[1:5:50, 2]
e = bs[1:5:50, 1]

n_sim = 50
σ_ss_params = [0.0, -4.0]#collect(-1.5:0.05:0.0)
σ_params = collect(4:2:10) .+ 0.001

n_sim*length(σ_ss_params)*length(σ_params)

#results = CSV.read("results_nlopt.csv", DataFrame)

results = DataFrame(σ = Float64[],
                    σ_ss = Float64[],
                    n_default = Int64[],
                    degree = Float64[],
                    interm = Int64[],
                    eq_r_l = Float64[],
                    mean_liq = Float64[],
                    mean_ib_share = Float64[],
                    mean_eq_req = Float64[],
                    mean_n_share = Float64[])

for σ_ss in σ_ss_params
    for σ in σ_params
        for sim in 1:n_sim
            seed = rand(1:10000000000)
            Random.seed!(seed)
            #println("seed: $seed | sim: $sim / $n_sim | σ = $σ / $(σ_params[end]) | σ_ss = $σ_ss / $(maximum(σ_ss_params))")
            bank_sys = BankSystem(α = 0.01,
                                    ω_n = 1.0, 
                                    ω_l = 0.6, 
                                    γ = 0.06,
                                    τ = 0.02, 
                                    ζ = 0.5, 
                                    exp_δ = 0.01, 
                                    σ_δ = 0.01)
            
            populate!(bank_sys, 
                        N = length(d), 
                        r_n = rand(Uniform(0.0, 0.1), length(d)), 
                        σ = rand([σ], length(d)),
                        d = d,
                        e = e)   

            super_spreader!(bank_sys, σ_ss)
            equilibrium!(bank_sys, verbose = false, min_iter = 20)      

            if maximum(balance_check(bank_sys)) > 0.001
                @warn "balance sheet identity not satisfied"
                continue
            end                

            println("max BS diff: ", maximum(balance_check(bank_sys)), " | imbalance: $(round(get_market_balance(bank_sys)))")
            adjust_imbalance!(bank_sys)
            try
                fund_matching!(bank_sys, 0.3)    
            catch
                try
                    fund_matching!(bank_sys, 0.5)  
                catch
                    @warn "NO fund_matching solution!"
                    continue
                end
            end
            
            res_sim = DataFrame(σ = [σ],
                                σ_ss = [σ_ss],
                                n_default = [0],
                                degree = [degree(bank_sys)],
                                interm = [length(intermediators(bank_sys))],
                                eq_r_l = [bank_sys.r_l],
                                mean_liq = [mean(liquidity(bank_sys))],
                                mean_ib_share = [mean(ib_share(bank_sys))],
                                mean_eq_req = [mean(equity_requirement(bank_sys))],
                                mean_n_share = [mean(bank.n / assets(bank, bank_sys) for bank in bank_sys.banks)])
                      
            contagion_liq!(bank_sys)

            res_sim.n_default[1] = n_default(bank_sys)
            results = [results; res_sim]
        end
    end
end

heatmap(σ_ss_params, σ_params, Matrix(heatmap_df)[:,2:end])

heatmap_df = @chain results begin
    groupby([:σ, :σ_ss])
    combine(:n_default => mean)
    sort()
    unstack(:σ, :n_default_mean)
end

CSV.write("data/results_lrisk.csv", results)

quantile(results.n_default, collect(0.5:0.1:1.0))

heatmap(σ_ss_params, σ_params, Matrix(heatmap_df)[:,2:end])

@chain results begin
    groupby([:σ_ss, :σ])
    combine(:n_default => mean)
    sort()
    unstack(:σ, :n_default_mean)
end    

@chain results begin
    groupby(:σ)
    combine([:n_default, :interm, :degree, :eq_r_l] .=> mean)
    #combine([:mean_n_share, :mean_liq, :mean_ib_share] .=> mean)
    sort()
end    

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
    groupby([:σ, :σ_ss])
    combine(:n_default => x -> sum(x .> 2)/sum(x .> 0))
    #combine(nrow => :count)
    sort()
    unstack(:σ, :n_default_function)
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



function contagion(N, α, ω_n, ω_l, γ, τ, d, e, σ, ζ, exp_δ, σ_δ, r_n, σ_rn)

    params = equilibrium(N, d, e, α, ω_n, ω_l, γ, τ, σ, ζ, exp_δ, σ_δ, r_n, σ_rn, N*2)

    eq_r_l = params.r_l[findmin(abs.(params.imbalance))[2]-1]
        
    optim_vars = zeros(N, 5)
    
    # c, n, l, b
    for bank in 1:N 
        println("liczenie #", bank, " banku")
        optim_vars[bank, 1:4] .= optim_allocation(d[bank], α, ω_n, ω_l, γ, τ, e[bank], r_n[bank], eq_r_l, ζ, exp_δ, σ_rn, σ_δ, σ[bank])
        optim_vars[bank, 5] = profit(r_n[bank], optim_vars[bank, 2], eq_r_l, optim_vars[bank, 3], ζ, exp_δ, optim_vars[bank, 4], optim_vars[bank, 1])
    end
 
    #length(findall(-0.001 .< balance_check(optim_vars[:,1], optim_vars[:,2], optim_vars[:, 3], d, optim_vars[:,4], e) .> 0.001)) > 0 && error("non-feasible")
    
    # imbalance adjustment 
    imbalance = sum(optim_vars[:,3]) - sum(optim_vars[:,4])
    
    println("any banks with not enoug liq: ", round.((optim_vars[:,1] .- (α .* d))))
    println("imbalance: ", round.(sum(optim_vars[:,3]) - sum(optim_vars[:,4])))

    if imbalance > 0
        borrowers = findall(optim_vars[:,4] .> 0.0001)
        optim_vars[borrowers, 4] .+= imbalance ./ length(borrowers)
        optim_vars[borrowers, 1] .+= imbalance ./ length(borrowers)
    elseif imbalance < 0
        lenders = findall(optim_vars[:, 3] .> 0.0001)
        optim_vars[lenders, 3] .-= imbalance ./ length(lenders)
        d[lenders] .-= imbalance ./ length(lenders)
    end    
    
    # l == b
    sum(optim_vars[:, 3]) - sum(optim_vars[:, 4])
    
    k = @. equity_requirement(optim_vars[:, 1], optim_vars[:, 2], optim_vars[:, 3], d, optim_vars[:, 4], ω_n, ω_l)
    A = optim_vars[:,1] .+ optim_vars[:,2] .+ optim_vars[:,3]
    
    println("min(l,b): ", round.(min(optim_vars[:, 3], optim_vars[:, 4])))

    A_ib = fund_matching(optim_vars[:, 3], optim_vars[:, 4], σ, 1 ./ (d ./ A), A, 0.2)
    
    #any(-0.001 .< balance_check(optim_vars[:,1], optim_vars[:,2], sum(A_ib, dims=2), d, sum(A_ib, dims=1)', e) .> 0.001)

    println("any with wrong BS: ", round.(balance_check(optim_vars[:,1], optim_vars[:,2], sum(A_ib, dims=2), d, sum(A_ib, dims=1)', e)))
    
    println("any bank with not enough cap: ", round.(k .- (γ + τ)))
    
    println("any bank with value < 0?: ", any(optim_vars .< 0))

    sum(sum(A_ib, dims=2) .- optim_vars[:, 3])
    sum(sum(A_ib, dims=1) .- optim_vars[:, 4])
    
    sum(clearing_vector(A_ib, optim_vars[:,1], optim_vars[:,2]) .- sum(A_ib, dims=1)' .< 0)
    
    results = zeros(9)
    results[2] = eq_r_l
    results[3] = mean(optim_vars[:,1] ./ d)
    results[4] = mean(round.(optim_vars[:, 3]))
    results[5] = mean(optim_vars[:, 5] ./ A)
    results[6] = mean(optim_vars[:,3] ./ A)
    results[7] = mean(k)
    results[8] = mean(e ./ A)
    results[9] = mean(optim_vars[:,2] ./ A)

    shocked_bank = rand(1:N,1)
    e[shocked_bank] .= 0
    optim_vars[shocked_bank, 2] .= 0
    #optim_vars[shocked_bank, 1] .= 0
    
    
    #(c + n + l) - (d + b + e)
    
    # OD TEGO MOMENTU 3 I 4 W optim_vars sa depreciated, uzywaj A_ib a jak chcesz zagregowane watosci to:
    # l = sum(A_ib, dims=2) = optim_vars[:, 3]
    # b = sum(A_ib, dims=1) = optim_vars[:, 4]
    
    
    # co jak bank upadł, calluje debt ale debtors nie maja wystarczajaco hajsu?
    e_t = [0, sum(e)]
    # clearing loop until there is no defaulted bank with any liquid assets (cash and IB)
    #  any(e .<= 0 .&& (sum(A_ib, dims=2) .> 0 .|| optim_vars[:, 1] .> 0))
    while e_t[end-1] != e_t[end]
    
        # identify defaults (negative capital)
        defaults = findall(e .<= 0)
        
        d[defaults] .-= optim_vars[defaults, 1] 
        optim_vars[defaults, 1] .= 0
        calling_banks = findall(e .<= 0 .&& (sum(A_ib, dims=2) .> 0 .|| optim_vars[:, 1] .> 0)[:])
    
        # stage 1: calling for liquidity
        for call_id in calling_banks
    
            debtors = findall(A_ib[call_id, :] .> 0)
            # debtors repay either what they owe or have
            d[call_id] += sum(min.(A_ib[call_id, :], optim_vars[:,1])) # + cash
            repayment = min.(A_ib[call_id, :], optim_vars[:,1])
            optim_vars[:,1] .-= repayment
            A_ib[call_id,:] .-= repayment # -IB assets
    
        end
    
        # stage 2: writing down remaining IB liabilities
        e .-= sum(A_ib[:, defaults], dims = 2)
        A_ib[:, defaults] .= 0
        push!(e_t, sum(e))
    end
    
    results[1] = sum(e .<= 0)

    # n_default | eq_r_l | mean liq | sd liq | mean A_ib / A | sd A_ib / A | mean eq_req | sd eq_req | length(e_t)
    return results
end



results = zeros(10)'
#results = deepcopy(results_extr)
n_sim = 1
σ_params = [0.0, 1.0, 2.0]
Random.seed!(12)


@time begin
    σ_params .+= 0.001
    results = zeros(10)'

    n_sim = 1
    σ_params = [1.01]
    N     = 20 # n banks
    α     = 0.01 # liquidity requirement
    ω_n   = 1 # risk weight on non-liquid assets
    ω_l   = 0.2 # risk weight on liquid assets
    γ     = 0.08 # equity requirement ratio
    τ     = 0.01 # equity buffer
    d     = [(606/1.06), (807/1.5), (529/1.08), (211/0.7), (838/1.47), (296/0.63), (250/0.68), (428/2), (284/1.24), (40/0.94), (8.2/0.2), (252/1.74), (24/0.19), (111.1/1.03), (88.9/1.3), (51.8/0.42), (63/0.48), (111.1/1.65), (100/1.37), (11.6/0.15)] # rand(Normal(700, 100), N) # deposits
    e     = [55.6, 90.0, 48.5, 53.0, 81.0, 53.0, 57.0, 48.0, 26.0, 43.0, 20.0, 23.0, 16.0, 10.0, 8.0, 5.0, 6.0, 10.0, 9.0, 9.0] #rand(Normal(50, 20), N) # equity
    #d     = rand(Normal(500, 50), N)
    #e     = rand(Normal(50, 5), N)
    σ     = rand([2.001], N) #rand(Uniform(2 - σ_sim, 2 + σ_sim), N) # risk aversion
    extreme = rand(1:N, 1)
    ζ     = 0.6 # lgd
    exp_δ = 0.005 # pd
    σ_δ   = 0.003 # variance of pd
    r_n   = rand(Uniform(0.01, 0.15), N) # return on non liquid assets
    σ_rn  = (1/12).*(maximum(r_n) - minimum(r_n)).^2

    for σ_sim in σ_params

        σ[extreme] .-= σ_sim
        σ[1:N .∉ Ref(extreme)] .+= σ_sim/(N-1)

        for i in  1:n_sim
            print("symulacja n: ", i, " | σ = ", σ_sim)
            results = vcat(results, [σ_sim contagion(N, α, ω_n, ω_l, γ, τ, d, e, σ, ζ, exp_δ, σ_δ, r_n, σ_rn)'])
        end 
    end
end


#184.735280

[[mean(results[findall(results_extr[:,1] .== σ_param), var]) for σ_param in unique(results[:,1])] for var in 1:10]
[[mean(results[findall(results[:,1] .== σ_param), var]) for σ_param in unique(results[:,1])] for var in 1:10]



[[mean(results[findall((results[:,1] .== σ_param) .& (results[:,2] .> 1)), var]) for σ_param in unique(results[:,1])] for var in 1:10]
[[length(results[findall((results[:,1] .== σ_param) .& (results[:,2] .> 1)), var]) for σ_param in unique(results[:,1])] for var in 1:10]


reshape([results_extr[findall(results_extr[:,1] .== σ_params[i]), 2] for i in 1:4])

results_extr[findall(results_extr[:,1] .== σ_params[1]), 2]

boxplot(results_extr[findall(results_extr[:,1] .== σ_params[1]), 2])

box_df = hcat(results_extr[findall(results_extr[:,1] .== σ_params[1]), 2], 
results_extr[findall(results_extr[:,1] .== σ_params[2]), 2],
results_extr[findall(results_extr[:,1] .== σ_params[3]), 2],
results_extr[findall(results_extr[:,1] .== σ_params[4]), 2])

a = rand(1:5, 100)
b = rand(1:5, 100)
c = randn(100)
using StatsPlots

boxplot(box_df)

as = rand(["a", "b", "c"], 100)
bs = randn(100)
boxplot(as, bs)

boxplot(results_extr[findall((results_extr[:,1] .!= 0.0) .& (results_extr[:,2] .!= 1.0)), 1],
        results_extr[findall((results_extr[:,1] .!= 0.0) .& (results_extr[:,2] .!= 1.0)), 2])

2+2

groupedboxplot()
groupedboxplot(results_extr[:,2], results_extr[:,1], bar_width = 0.8)
unique(results_extr[:,1])

hcat(results_extr[findall(results_extr[:,1] .== σ_params[i]), 2] for i in 1:4)
histogram([results_extr[findall(results_extr[:,1] .== σ_params[4]), 2]])


results_extr = readdlm("results.csv", ',', Float64)

writedlm( "results.csv",  results, ',')

2000/60

unique(results[:,1])

[length(results_extr[findall(results_extr[:,1] .== σ_param), 2]) for σ_param in unique(results[:,1])] # n obserwacji per grupa
[[maximum(results[findall(results[:,1] .== σ_param), var]) for σ_param in σ_params] for var in 1:10]
[[minimum(results[findall(results[:,1] .== σ_param), var]) for σ_param in σ_params] for var in 1:10]

# dlaczego im wieksze risk aersion tym mniejsze capital adequacy???
# 

#results_unif = deepcopy(results)
#10
# σ |  n_default | eq_r_l | mean liq | mean l | mean A_ib / A | mean(l / A) | mean eq_req | sd eq_req | mean(n / A)

mean(results_extr[:,3])

results[5] = mean(optim_vars[:, 5] ./ A)
results[6] = mean(optim_vars[:,3] ./ A)
results[7] = mean(k)
results[8] = std(k)
results[9] = mean(optim_vars[:,2] ./ A)

params = equilibrium(N, d, e, α, ω_n, ω_l, γ, τ, σ, ζ, exp_δ, σ_δ, r_n, σ_rn, N*2)

eq_r_l = params.r_l[findmin(abs.(params.imbalance))[2]-1]

mean(e ./ d)