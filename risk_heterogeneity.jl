using Random
using Distributions
#using Plots
using LinearAlgebra
using JuMP
using Ipopt
using NLopt
using HiGHS


function exp_utility(exp_profit, σ_profit, σ)
    return ((exp_profit)^(1-σ))/(1 - σ) - (((σ/2)*(exp_profit)^(-(1+σ))) * σ_profit)
    #return (1/σ) - (1/σ) * exp(-σ*exp_profit) - (1/2) * σ * exp(-exp_profit * σ) * σ_profit
end

function profit(r_n, n, r_l, l, ζ, δ, b, c)
    return (r_n * n + r_l * l) - ((1 /(1 - ζ * δ)) *r_l * b) + c * 0.02
end

function σ_profit(n, σ_rn, b, r_l, ζ, σ_δ, exp_δ)
    return n^2 * σ_rn - (b * r_l)^2 * ζ^2 * (1 - (ζ * exp_δ))^(-4) * σ_δ
end

function balance_check(c, n, l, d, b, e)
    return (c + n + l) - (d + b + e)
end
 
function equity_requirement(c, n, l, d, b, ω_n, ω_l)
    return (c + n + l - d - b)/(ω_n * n + ω_l * l)
end

function optim_allocation_msg(model)

    if primal_status(model) == FEASIBLE_POINT
        error("No feasible primal")
    end
    if  dual_status(model) == FEASIBLE_POINT
        error("No feasible dual")
    end

    if termination_status(model) == OPTIMAL
        println("Solution is optimal")
    elseif termination_status(model) == TIME_LIMIT && has_values(model)
        println("Solution is suboptimal due to a time limit, but a primal solution is available")
    elseif primal_status(model) != FEASIBLE_POINT
        error("no ladnie")
    end
    println(" objective value = ", objective_value(model), " | ", primal_status(model), " | ", termination_status(model))
end

function optim_allocation(d, α, ω_n, ω_l, γ, τ, e, r_n, r_l, ζ, exp_δ, σ_rn, σ_δ, σ)

    #allocation = Model(Ipopt.Optimizer)
    allocation = Model(NLopt.Optimizer)
    set_optimizer_attribute(allocation, "algorithm", :LN_COBYLA) #:LN_COBYLA 
    set_optimizer_attribute(allocation, "maxtime", 10)
    set_silent(allocation)
    register(allocation, :σ_profit, 7, σ_profit; autodiff = true)
    register(allocation, :exp_utility, 3, exp_utility; autodiff = true)
    register(allocation, :profit, 8, profit; autodiff = true)
    register(allocation, :equity_requirement, 7, equity_requirement; autodiff = true)
    @variable(allocation, c >= 0.1)
    @variable(allocation, n >= 0.1)
    @variable(allocation, l >= 0.1)
    @variable(allocation, b >= 0.1)

    # balance sheet identity
    @constraint(allocation, balance_check(c, n, l, d, b, e) == 0)

    # liquidity requirement
    @constraint(allocation, c >= α * d)

    # capital requirement
    @constraint(allocation, c + l + n >= b + d + (γ + τ) * (ω_n * n + ω_l * l))
    #@NLconstraint(allocation, (c + n + l - d - b) >= (γ + τ) * (ω_n * n + ω_l * l))
    #@NLconstraint(allocation, equity_requirement(c, n, l, d, b, ω_n, ω_l) >= (γ + τ))
    #(c + n + l - d - b)/(ω_n * n + ω_l * l)
    
    if σ == 0.00
        @NLobjective(allocation, Max, profit(r_n, n, r_l, l, ζ, exp_δ, b, c))
    else
        # positive profit
        @NLconstraint(allocation, profit(r_n, n, r_l, l, ζ, exp_δ, b, c) >= 0.1)
        @NLobjective(allocation, Max, exp_utility(profit(r_n, n, r_l, l, ζ, exp_δ, b, c), σ_profit(n, σ_rn, b, r_l, ζ,σ_δ, exp_δ), σ))
    end
    

    JuMP.optimize!(allocation)

#   optim_allocation_msg(allocation)
#    primal_status(allocation) != FEASIBLE_POINT && error("no feasible primal", solution_summary(allocation))

    return [value(dec_var) for dec_var in [c, n, l, b]]
end


function get_market_balance(N, d, e, α, ω_n, ω_l, γ, τ, σ, ζ, exp_δ, σ_δ, r_n, σ_rn, r_l)

    optim_vars = zeros(N, 4)

    for bank in 1:N 
        print("|", bank, "|")
        optim_vars[bank, 1:4] .= optim_allocation(d[bank], α, ω_n, ω_l, γ, τ, e[bank], r_n[bank], r_l, ζ, exp_δ, σ_rn, σ_δ, σ[bank])
    end

    return sum(optim_vars[:,3]) - sum(optim_vars[:, 4])
end    


function equilibrium(N, d, e, α, ω_n, ω_l, γ, τ, σ, ζ, exp_δ, σ_δ, r_n, σ_rn, tol)
    
    param_space = (r_l = [0.05, 0.05], imbalance = [10000, Inf], up_bound = [0.1], low_bound = [0.0])
    
    while (abs(param_space.imbalance[end]) > tol) && ((abs(param_space.imbalance[end] - param_space.imbalance[end-1]) > 1) | length(param_space.imbalance) < 20)
        
        println("\n iteracja: r_l: ", param_space.r_l[end], " | imbalans:", param_space.imbalance[end])
        push!(param_space.imbalance, get_market_balance(N, d, e, α, ω_n, ω_l, γ, τ, σ, ζ, exp_δ, σ_δ, r_n, σ_rn, param_space.r_l[end]))
        
        # if too much supply, choose next r_l is a midpoint between last and minimum of previous 3 r_l (halving r_l)
        if param_space.imbalance[end] > 0
            push!(param_space.up_bound, param_space.r_l[end])
            push!(param_space.r_l, (param_space.r_l[end] + param_space.low_bound[end])/2)            
        else
            push!(param_space.low_bound, param_space.r_l[end])
            push!(param_space.r_l, (param_space.r_l[end] + param_space.up_bound[end])/2)
        end
    end

    return param_space
end

function fund_matching(l, b, σ, k, A, max_expo)
    
    N = size(l)[1]
    fund_matching_optim = Model(HiGHS.Optimizer)
    set_silent(fund_matching_optim)
    @variable(fund_matching_optim, A_ib[1:N, 1:N] >= 0)    

    # borrowers constraint
    for i in 1:N
        @constraint(fund_matching_optim, sum(A_ib[:,i]) == b[i])
    end
        
    # savers constraint
    for i in 1:N 
        @constraint(fund_matching_optim, sum(A_ib[i,:]) == l[i])
    end

    # no self trading
    for i in 1:N
        @constraint(fund_matching_optim, A_ib[i,i] == 0)
    end

    for i in 1:N 
        @constraint(fund_matching_optim, A_ib[:, i] ./ A[i] .<= max_expo)  
    end

    capital_rate = k ./ A

    #@objective(fund_matching_optim, Max,  sum(σ[i] * (A_ib[i,:]'capital_rate) for i in 1:N))
    @objective(fund_matching_optim, Max,  sum(σ[i] * (A_ib[i,:]'k) for i in 1:N))
    JuMP.optimize!(fund_matching_optim)

    return value.(A_ib)
end

function clearing_vector(A_ib, c, n)
    
    function get_asset_share(A_ib, i)
        share = ([A_ib[i, k] / sum(A_ib[:,k]) for k in 1:N]) # calculate bank i share of total assets of other banks
        share = ifelse.(isnan.(share), 0, share) # if one bank dont have liabilities, then it's 0 (gives NaN if x/0)
        return share
    end

    p̄ = sum(A_ib, dims=1)'

    N = size(p̄)[1]
    market_clearing = Model(HiGHS.Optimizer)
    set_silent(market_clearing)
    @variable(market_clearing, p[1:N] >= 0)    

    # borrowers constraint
    # cant pay more than what is owed
    for i in 1:N
        @constraint(market_clearing, p[i] <= p̄[i])
    end

    # p_i ≤ c_i + sum(p * a_i)
    # cant pay more than amount of assets at hand + amount of available IB assets
    for i in 1:N
        @constraint(market_clearing, p[i] <= c[i] + n[i] + sum(p .* get_asset_share(A_ib, i)))
    end

    @objective(market_clearing, Max,  sum(p))
    JuMP.optimize!(market_clearing)

    return value.(p)
end

