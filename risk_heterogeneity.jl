using Pkg
#Pkg.add.(["Distributions", "MarSwitching", "JuMP", "Ipopt", "NLopt", "HiGHS"])
using Random
using Distributions
#using Plots
using LinearAlgebra
using JuMP
using Ipopt
using NLopt
using HiGHS
using Printf
using ForwardDiff

mutable struct Bank{V <: AbstractFloat}
    const id::Int64
    const r_n::V
    # balance sheet
    c::V
    n::V
    l::V
    d::V
    e::V
    b::V
    # preference params
    const σ::V
    const σ_rn::V
end

Base.@kwdef mutable struct BankSystem{V <: AbstractFloat}
    # regulatory params
    const α::V
    const ω_n::V
    const ω_l::V
    const γ::V
    const τ::V
    const ζ::V
    const exp_δ::V
    const σ_δ::V
    banks::Vector{Bank} = Bank[]
    r_l::V = 0.0
    A_ib::Matrix{V} = Matrix{Float64}(undef, 0, 0)
end    

# show table of c, n, l, b of eery bank with rounding
function Base.show(io::IO, ::MIME"text/plain", banks::Vector{Bank})
    print(io, "Bank | c | n | l | b | e\n")
    for bank in banks
        print(io, bank.id, " | ", round(bank.c), " | ", round(bank.n), " | ", round(bank.l), " | ", round(bank.b), " | ", round(bank.e), "\n")
    end
end    

function print_bs(banks::Vector{Bank})
    print("Bank | c | n | l | b | e\n")
    for bank in banks
        print(bank.id, " | ", round(bank.c), " | ", round(bank.n), " | ", round(bank.l), " | ", round(bank.b), " | ", round(bank.e), "\n")
    end
end    


exp_utility(exp_profit, σ_profit, σ) = ((exp_profit)^(1-σ))/(1 - σ) - (((σ/2)*(exp_profit)^(-(1+σ))) * σ_profit)

function exp_utility(bank::Bank, bank_system::BankSystem)
    exp_profit = profit(bank, bank_system)
    σ_profit = σ_profit(bank, bank_system)
    return ((exp_profit)^(1-bank.σ))/(1 - bank.σ) - (((bank.σ/2)*(exp_profit)^(-(1+bank.σ))) * σ_profit)
    #return (1/σ) - (1/σ) * exp(-σ*exp_profit) - (1/2) * σ * exp(-exp_profit * σ) * σ_profit
end

n_default(bank_sys::BankSystem) = length(findall(x -> x.e <= 0, bank_sys.banks))

profit(r_n, n, r_l, l, ζ, δ, b) = (r_n * n + r_l * l) - ((1 /(1 - ζ * δ)) *r_l * b)
profit(bank::Bank, bank_system::BankSystem) = (bank.r_n * bank.n + bank_system.r_l * bank.l) - ((1 /(1 - bank_system.ζ * bank_system.exp_δ)) * bank_system.r_l * bank.b)  

σ_profit(n, σ_rn, b, r_l, ζ, σ_δ, exp_δ) =  n^2 * σ_rn - (b * r_l)^2 * ζ^2 * (1 - (ζ * exp_δ))^(-4) * σ_δ
σ_profit(bank::Bank, bank_system::BankSystem) = bank.n^2 * bank.σ_rn - (bank.b * bank_system.r_l)^2 * bank_system.ζ^2 * (1 - (bank_system.ζ * bank_system.exp_δ))^(-4) * bank_system.σ_δ

balance_check(c, n, l, d, b, e) = (c + n + l) - (d + b + e)
balance_check(bank::Bank) = (bank.c + bank.n + bank.l) - (bank.d + bank.b + bank.e)

degree(bank_sys::BankSystem) = mean(sum(bank_sys.A_ib .> 0, dims = 1))
assets(bank_sys::BankSystem) = [bank.c + bank.n + bank.l for bank in bank_sys.banks]
assets(bank::Bank) = bank.c + bank.n + bank.l

ib_share(bank_sys::BankSystem) = (sum(bank_sys.A_ib)/2) / sum(assets(bank_sys))
leverage(bank_sys::BankSystem) = [bank.e / assets(bank) for bank in bank_sys.banks]
leverage(bank::Bank) = bank.e / assets(bank)

liquidity(bank::Bank) = bank.c / bank.d
liquidity(bank_sys::BankSystem) = [bank.c / bank.d for bank in bank_sys.banks]

intermediators(bank_sys::BankSystem) = findall(x -> (x.l > 2) & (x.b > 2), bank_sys.banks)

equity_requirement(c, n, l, d, b, ω_n, ω_l) = (c + n + l - d - b)/(ω_n * n + ω_l * l)
equity_requirement(bank::Bank, bank_system::BankSystem) = (bank.c + bank.n + bank.l - bank.d - bank.b)/(bank_system.ω_n * bank.n + bank_system.ω_l * bank.l)
equity_requirement(bank_system::BankSystem) = [(bank.c + bank.n + bank.l - bank.d - bank.b)/(bank_system.ω_n * bank.n + bank_system.ω_l * bank.l) for bank in bank_system.banks]


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


function populate!(bank_sys::BankSystem; 
                   N = 20, 
                   d = Vector{Float64}([]), 
                   e = Vector{Float64}([]), 
                   σ = Vector{Float64}([]), 
                   r_n = Vector{Float64}([]))

    d = isempty(d) ? rand(Normal(500, 80), N) : d
    e = isempty(e) ? rand(Normal(90, 10), N) : e
    σ = isempty(σ) ? rand([1.5], N) : σ
    r_n = isempty(r_n) ? rand(Uniform(0.03, 0.15), N) : r_n
    σ_rn  = (1/12).*(maximum(r_n) - minimum(r_n)).^2

    for i in 1:N
        push!(bank_sys.banks, Bank(i, r_n[i], 0.0, 0.0, 0.0, d[i], e[i], 0.0, σ[i], σ_rn))
    end
end

function get_market_balance(bank_sys::BankSystem)

    agg_demand = 0.0
    agg_supply = 0.0

    for bank in bank_sys.banks
        agg_demand += bank.b
        agg_supply += bank.l
    end

    return  agg_supply - agg_demand 
end

function get_market_balance(N, d, e, α, ω_n, ω_l, γ, τ, σ, ζ, exp_δ, σ_δ, r_n, σ_rn, r_l)

    optim_vars = zeros(N, 4)

    for bank in 1:N 
        print("|", bank, "|")
        optim_vars[bank, 1:4] .= optim_allocation(d[bank], α, ω_n, ω_l, γ, τ, e[bank], r_n[bank], r_l, ζ, exp_δ, σ_rn, σ_δ, σ[bank])
    end

    return sum(optim_vars[:,3]) - sum(optim_vars[:, 4])
end    

function equilibrium!(bank_sys::BankSystem; tol = -1.0, min_iter = 20, verbose = true)

    tol = tol < 0 ? length(bank_sys.banks)*2 : tol

    params = (r_l = [0.05, 0.05], diff = [10000, Inf], up_bound = [0.1], low_bound = [0.0])

    while (abs(params.diff[end]) > tol) && ((abs(params.diff[end] - params.diff[end-1]) > 1) | length(params.diff) < min_iter)
        verbose && println("\n iteration: r_l: ", params.r_l[end], " | imbalance: ", round(params.diff[end])) 
        bank_sys.r_l = params.r_l[end]

        # optim_allocation!.(bank_sys.banks, bank_sys) ?
        for bank in bank_sys.banks
            verbose && print("|", bank.id, "|")
            optim_allocation!(bank, bank_sys)
        end

        push!(params.diff, get_market_balance(bank_sys))
        
         # if too much supply, choose next r_l is a midpoint between last and minimum of previous 3 r_l (halving r_l)
        if params.diff[end] > 0
            push!(params.up_bound, params.r_l[end])
            push!(params.r_l, (params.r_l[end] + params.low_bound[end])/2)            
        else
            push!(params.low_bound, params.r_l[end])
            push!(params.r_l, (params.r_l[end] + params.up_bound[end])/2)
        end
    end

    params.r_l[findmin(abs.(params.diff))[2]-1]
    bank_sys.r_l = params.r_l[findmin(abs.(params.diff))[2]-1]

    # optim_allocation!.(bank_sys.banks, bank_sys) ?
    for bank in bank_sys.banks
        optim_allocation!(bank, bank_sys)
    end
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

function fund_matching!(bank_sys::BankSystem, max_expo = 0.1)
    bank_sys.A_ib = fund_matching([bank.l for bank in bank_sys.banks],
                                  [bank.b for bank in bank_sys.banks],
                                  [bank.σ for bank in bank_sys.banks], 
                                  equity_requirement(bank_sys), 
                                  [bank.c + bank.n + bank.l for bank in bank_sys.banks],
                                  max_expo)
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
    
    #@objective(fund_matching_optim, Max,  sum(σ[i] * (A_ib[i,:]'capital_rate) for i in 1:N))
    @objective(fund_matching_optim, Max,  sum(σ[i] * (A_ib[i,:]'k) for i in 1:N))
    JuMP.optimize!(fund_matching_optim)

    return value.(A_ib)
end

function adjust_imbalance!(bank_sys)
    imbalance = get_market_balance(bank_sys)

    println("max liquidity shortfall: $(minimum(round.([bank.c - (bank_sys.α * bank.d) for bank in bank_sys.banks])))")
    println("imbalance: $(round.(imbalance))")
    
    if imbalance > 0
        borrowers = findall([bank.b for bank in bank_sys.banks] .> 1)
        [bank.b += imbalance / length(borrowers) for bank in bank_sys.banks[borrowers]]
        [bank.c += imbalance / length(borrowers) for bank in bank_sys.banks[borrowers]]    
    elseif imbalance < 0
        lenders = findall([bank.l for bank in bank_sys.banks] .> 1)
        [bank.l -= imbalance / length(lenders) for bank in bank_sys.banks[lenders]]
        [bank.d -= imbalance / length(lenders) for bank in bank_sys.banks[lenders]]
    end 
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

function contagion!(bank_sys::BankSystem)
    
    # shock
    N = length(bank_sys.banks)
    shocked_bank = rand(1:N)

    # writing down shock
    bank_sys.banks[shocked_bank].e = 0
    bank_sys.banks[shocked_bank].n = 0

    e_t = [0, sum([bank.e for bank in bank_sys.banks])]

    while e_t[end-1] != e_t[end]

        # identify defaults (negative capital)
        defaults = findall(x -> x.e <= 0, bank_sys.banks)
        
        # repaying deposits with cash
        for default in defaults
            bank_sys.banks[default].d -= bank_sys.banks[default].c
            bank_sys.banks[default].c = 0
        end

        # which bank needs liquidity?
        calling_banks = findall(x -> x.e <= 0 && (x.l > 0 || x.c > 0), bank_sys.banks)
        
        # calling for liquidity
        for call_id in calling_banks
            
            debtors = findall(bank_sys.A_ib[call_id, :] .> 0) # debtors of calling bank
            
            for debtor in debtors
                repayment = min.(bank_sys.A_ib[call_id, debtor], bank_sys.banks[debtor].c)
                bank_sys.banks[call_id].c      += repayment
                bank_sys.banks[debtor].c       -= repayment
                bank_sys.A_ib[call_id, debtor] -= repayment
                bank_sys.banks[call_id].l      -= repayment
                bank_sys.banks[debtor].b       -= repayment
            end            
        end

        #  writing down remaining IB liabilities
        for default in defaults
            creditors = findall(bank_sys.A_ib[:, default] .> 0)
            for creditor in creditors
                bank_sys.banks[creditor].e -= bank_sys.A_ib[creditor, default]
                bank_sys.A_ib[creditor, default] = 0
            end
        end    
        push!(e_t, sum([bank.e for bank in bank_sys.banks]))
    end
end
