using Random
using Distributions
using Plots
using LinearAlgebra
using JuMP
import HiGHS

function sum_balance(A_ib, class)
    n = size(A_ib)[1]
    if class == "assets"
        return [sum(A_ib[i, 1:n]) for i in 1:n]
    else
        return [sum(A_ib[1:n, i]) for i in 1:n]
    end    
end

function degree(bank, A_ib, direction)
    n = size(A_ib)[1]
    if direction == "in"
        return sum(A_ib[bank, 1:n] .!= 0)
    else
        return sum(A_ib[1:n, bank] .!= 0)
    end
end

function capital_buffer(bank, A_ib, D, A_M)
    return sum_balance(A_ib, "assets")[bank] .+ A_M[bank] - sum_balance(A_ib, "liab")[bank] - D[bank]
end

function funds_matching(Δ, C, A_M, capital_ratio, γ)
    
    n = size(Δ)[1]
    risk = 1 ./ capital_ratio

    fund_matching = Model(HiGHS.Optimizer)
    set_silent(fund_matching)
    @variable(fund_matching, A_ib[1:n, 1:n] >= 0)

    # borrowers constraint
    for i in 1:n 
        @constraint(fund_matching, sum(A_ib[:,i]) <= Δ[i])
    end
        
    # savers constraint
    for i in 1:n 
        @constraint(fund_matching, sum(A_ib[i,:]) <= C[i])
    end

    # no self trading
    for i in 1:n 
        @constraint(fund_matching, A_ib[i,i] == 0)
    end

    @objective(fund_matching, Min, sum(A_ib[i,:]'A_ib[i,:] - γ[i] * (A_ib[i,:]'risk) for i in 1:n))
    optimize!(fund_matching)

    return value.(A_ib)
end


function contagion_optim(n, exp_A_M, σ_A_M, exp_buffer_rate, σ_buffer_rate, exp_Δ, σ_Δ, exp_C, σ_C, exp_γ, σ_γ)

    A_M = rand(Normal(exp_A_M,σ_A_M), n)
    buffer_rate = rand(Normal(exp_buffer_rate, σ_buffer_rate), n)
    Δ = clamp.(rand(Normal(exp_Δ, σ_Δ), n), 0, Inf) # loan demand 
    C = clamp.(rand(Normal(exp_C, σ_C), n), 0, Inf) # savings (cash)
    γ = clamp.(rand(Normal(exp_γ, σ_γ), n), 0.00001, Inf)
    D = ((A_M .+ C) .* (1 .- buffer_rate)) .- Δ
    k = (A_M .+ C) .- (D .+ Δ)
    defaults_t = [-1, 0]
    defaults = []
    capital_ratio = vcat((k ./ (A_M .+ C))', (k ./ (A_M .+ C))')
    
    while defaults_t[end] != defaults_t[end-1]

        A_ib = funds_matching(Δ, C, A_M, capital_ratio[end-1,:], γ)
        
        inbalance = (A_M .+ sum_balance(A_ib, "assets")) .- (D .+ sum_balance(A_ib, "liab") .+ k)
    
        D = D .- abs.(clamp.(inbalance, -Inf, 0)) # remove surplus of deposits (bank do not know what to do with them)
        A_M = A_M .- clamp.(inbalance, 0, Inf) # remove surplus of loans (bank ddo not know how to finance them)
    
        capital_ratio = vcat(capital_ratio,  (k ./ (A_M .+ sum_balance(A_ib, "assets")))')
    
        # if no defaults (first turn), then make a random default
        if size(defaults)[1] == 0
            push!(defaults, rand(findall(x -> x > 0, sum_balance(A_ib, "liab")), 1)[1])
        else
            defaults = unique(vcat(defaults, findall(capital_ratio[end, :] .< 0)))
        end
    
        A_ib[:, defaults] .= 0
        k = k .+ (A_M .+ sum_balance(A_ib, "assets")) .- (D .+ sum_balance(A_ib, "liab") .+ k)
        C = sum_balance(A_ib, "assets")
        C[defaults] .= 0
        k[defaults] .= -1
        
        push!(defaults_t, size(defaults)[1])
    
        Δ = sum_balance(A_ib, "liab")
        Δ[defaults] .= 0
        A_ib = zeros(n,n)
    
        capital_ratio = vcat(capital_ratio, (k ./ (A_M .+ sum_balance(A_ib, "assets") .+ C))')
    end

    return defaults_t
end

### simulations with different risk apetite heterogeneity ###

n = 20
exp_A_M = 90
σ_A_M = 0
exp_buffer_rate = 0.04
σ_buffer_rate = 0
exp_Δ = 20
σ_Δ = 40
exp_C = 20
σ_C = 40
exp_γ = 0.3
σ_γ = 0.1

sys_threshold = 0.1
n_runs = 300
results = []
Random.seed!(1)

# [0, 0.025, 0.05, 0.075, 0.1, 0.15]
for z in 0:0.01:0.1
    runs = []
    for contagion in 1:n_runs
        push!(runs, contagion_optim(n, exp_A_M, σ_A_M, exp_buffer_rate, σ_buffer_rate, exp_Δ, σ_Δ, exp_C, σ_C,exp_γ, z)[end])     
    end
    systematic = filter(x -> x > n * sys_threshold, runs)
    systematic = length(systematic) == 0 ? 0 : mean(systematic)
    push!(results, [z, systematic, sum(runs .> n * sys_threshold) / n_runs])
    println("calculating contagion with z = ", z, " | extent: ", systematic, " | default prob: ", sum(runs .> n * sys_threshold) / n_runs)
end


results = (reduce(hcat, results[1:end])')
results[:, 2] .= results[:, 2] ./ n

contagion_plot = plot(results[:,1], results[:,3],
  label="Frequency of contagion")
plot!(legend=:outerbottom, legendcolumns=2)

### simulations with different risk apetite ###

n = 20
exp_A_M = 80
σ_A_M = 0
exp_buffer_rate = 0.04
σ_buffer_rate = 0
exp_Δ = 20
σ_Δ = 40
exp_C = 20
σ_C = 40

sys_threshold = 0.1
n_runs = 300
results = []
Random.seed!(1)

# [0, 0.025, 0.05, 0.075, 0.1, 0.15]
for z in 0.1:0.05:1
    runs = []
    for contagion in 1:n_runs
        push!(runs, contagion_optim(n, exp_A_M, σ_A_M, exp_buffer_rate, σ_buffer_rate, exp_Δ, σ_Δ, exp_C, σ_C,z, 0)[end])     
    end
    systematic = filter(x -> x > n * sys_threshold, runs)
    systematic = length(systematic) == 0 ? 0 : mean(systematic)
    push!(results, [z, systematic, sum(runs .> n * sys_threshold) / n_runs])
    println("calculating contagion with z = ", z, " | extent: ", systematic, " | default prob: ", sum(runs .> n * sys_threshold) / n_runs)
end


results = (reduce(hcat, results[1:end])')
results[:, 2] .= results[:, 2] ./ n

contagion_plot = plot(results[:,1], results[:,2:3],
  label=["Extent of contagion" "Frequency of contagion"])
plot!(legend=:outerbottom, legendcolumns=2)

