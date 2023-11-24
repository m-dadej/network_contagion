    
function optim_allocation!(bank::Bank, bank_sys::BankSystem)
    
    allocation = Model(NLopt.Optimizer)
    set_optimizer_attribute(allocation, "algorithm", :LN_COBYLA) #:LN_COBYLA, LD_SLSQP
    set_optimizer_attribute(allocation, "maxtime", 10)
    set_silent(allocation)
    register(allocation, :σ_profit, 7, σ_profit; autodiff = true)
    register(allocation, :exp_utility, 3, exp_utility; autodiff = true)
    register(allocation, :profit, 7, profit; autodiff = true)
    register(allocation, :equity_requirement, 7, equity_requirement; autodiff = true)
    @variable(allocation, c >= 0.1)
    @variable(allocation, n >= 0.1)
    @variable(allocation, l >= 0.1)
    @variable(allocation, b >= 0.1)

    # balance sheet identity
    @constraint(allocation, balance_check(c, n, l, bank.d, b, bank.e) == 0)

    # liquidity requirement
    @constraint(allocation, c >= bank_sys.α * bank.d)

    # capital requirement
    @constraint(allocation, c + l + n >= b + bank.d + (bank_sys.γ + bank_sys.τ) * (bank_sys.ω_n * n + bank_sys.ω_l * l))
    #@NLconstraint(allocation, (c + n + l - d - b) >= (γ + τ) * (ω_n * n + ω_l * l))
    #@NLconstraint(allocation, equity_requirement(c, n, l, d, b, ω_n, ω_l) >= (γ + τ))
    #(c + n + l - d - b)/(ω_n * n + ω_l * l)
    
    if bank.σ == 0.00
        @NLobjective(allocation, Max, profit(bank.r_n, n, bank_sys.r_l, l, bank_sys.ζ, bank_sys.exp_δ, b))
    else
        # positive profit
        #@NLconstraint(allocation, profit(bank.r_n, n, bank_sys.r_l, l, bank_sys.ζ, bank_sys.exp_δ, b, c) >= 0.1)
        @NLobjective(allocation, Max, exp_utility(profit(bank.r_n, n, bank_sys.r_l, l, bank_sys.ζ, bank_sys.exp_δ, b), σ_profit(n, bank.σ_rn, b, bank_sys.r_l, bank_sys.ζ,bank_sys.σ_δ, bank_sys.exp_δ), bank.σ))
    end
     
    JuMP.optimize!(allocation)

    bank.c = value(c)
    bank.n = value(n)
    bank.l = value(l)
    bank.b = value(b)
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
    @constraint(allocation, e / (ω_n * n + ω_l * l) >= (γ + τ))
    #@constraint(allocation, c + l + n >= b + d + (γ + τ) * (ω_n * n + ω_l * l))
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

    return [value(dec_var) for dec_var in [c, n, l, b]]
end