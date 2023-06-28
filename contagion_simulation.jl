using DelimitedFiles
using CSV

function contagion(N, α, ω_n, ω_l, γ, τ, d, e, σ, ζ, exp_δ, σ_δ, r_n, σ_rn)

    params = equilibrium(N, d, e, α, ω_n, ω_l, γ, τ, σ, ζ, exp_δ, σ_δ, r_n, σ_rn, N*2)

    eq_r_l = params.r_l[findmin(abs.(params.imbalance))[2]-1]
        
    optim_vars = zeros(N, 5)
    
    # c, n, l, b
    for bank in 1:N 
        println("liczenie #", bank, " banku")
        optim_vars[bank, 1:4] .= optim_allocation(d[bank], α, ω_n, ω_l, γ, τ, e[bank], r_n[bank], eq_r_l, ζ, exp_δ, σ_rn, σ_δ, σ[bank])
        optim_vars[bank, 5] = profit(r_n[bank], optim_vars[bank, 2], eq_r_l, optim_vars[bank, 3], ζ, exp_δ, optim_vars[bank, 4])
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
    #d     = [(606/1.06), (807/1.5), (529/1.08), (211/0.7), (838/1.47), (296/0.63), (250/0.68), (428/2), (284/1.24), (40/0.94), (8.2/0.2), (252/1.74), (24/0.19), (111.1/1.03), (88.9/1.3), (51.8/0.42), (63/0.48), (111.1/1.65), (100/1.37), (11.6/0.15)] # rand(Normal(700, 100), N) # deposits
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

results[5] = mean(optim_vars[:, 5] ./ A)
results[6] = mean(optim_vars[:,3] ./ A)
results[7] = mean(k)
results[8] = std(k)
results[9] = mean(optim_vars[:,2] ./ A)

params = equilibrium(N, d, e, α, ω_n, ω_l, γ, τ, σ, ζ, exp_δ, σ_δ, r_n, σ_rn, N*2)

eq_r_l = params.r_l[findmin(abs.(params.imbalance))[2]-1]

mean(e ./ d)