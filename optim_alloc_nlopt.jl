function utility(x::AbstractVector{T}, bank::Bank, bank_sys::BankSystem) where T      
    prof = (bank.r_n * x[2] + bank_sys.r_l * x[3]) - ((1 /(1 - bank_sys.ζ * bank_sys.exp_δ)) * bank_sys.r_l * x[4])
    prof = max(prof, 0) # otherwise we have a domain error
    σ_prof = x[2]^2 * bank.σ_rn - (x[4] * bank_sys.r_l)^2 * bank_sys.ζ^2 * (1 - (bank_sys.ζ * bank_sys.exp_δ))^(-4) * bank_sys.σ_δ
    return prof^(1-bank.σ)/(1-bank.σ) - (((bank.σ/2)*(prof)^(-(1+bank.σ))) * σ_prof)
end

function obj_f(x::AbstractVector{T}, fΔ::AbstractVector{T},  bank::Bank, bank_sys::BankSystem) where T
    
    if length(fΔ) > 0
        fΔ[1:length(x)] .= ForwardDiff.gradient(x -> -utility(x, bank, bank_sys), x)
    end        
    
    return -utility(x, bank, bank_sys)
end 

function prof_inequality(x::AbstractVector{T}, fΔ_prof::AbstractVector{T}, bank::Bank, bank_sys::BankSystem) where T
    if length(fΔ_prof) > 0
        fΔ_prof[1:length(x)] .= ForwardDiff.gradient(x -> -((bank.r_n * x[2] + bank_sys.r_l * x[3]) - ((1 /(1 - bank_sys.ζ * bank_sys.exp_δ)) * bank_sys.r_l * x[4])) + 0.1, x)
    end

    return -((bank.r_n * x[2] + bank_sys.r_l * x[3]) - ((1 /(1 - bank_sys.ζ * bank_sys.exp_δ)) * bank_sys.r_l * x[4])) + 0.5
end

function bs_equality(x::AbstractVector{T}, fΔ_bs::AbstractVector{T}, bank::Bank) where T
    
    if length(fΔ_bs) > 0
        fΔ_bs[1:length(x)] .= ForwardDiff.gradient(x -> (x[1] + x[2] + x[3]) - (bank.d + x[4] + bank.e), x)
    end

    return (x[1] + x[2] + x[3]) - (bank.d + x[4] + bank.e)
end

function liq_inequality(x::AbstractVector{T}, fΔ_liq::AbstractVector{T}, bank::Bank, bank_sys::BankSystem) where T
    if length(fΔ_liq) > 0
        fΔ_liq[1:length(x)] .= ForwardDiff.gradient(x -> bank_sys.α * bank.d - x[1], x)
    end

    return bank_sys.α * bank.d - x[1]
end

function cap_inequality(x::AbstractVector{T}, fΔ_cap::AbstractVector{T}, bank::Bank, bank_sys::BankSystem) where T
    if length(fΔ_cap) > 0
        fΔ_cap[1:length(x)] .= ForwardDiff.gradient(x -> (bank_sys.γ + bank_sys.τ) - (((x[1] + x[2] + x[3]) - (bank.d + x[4])) /  (bank_sys.ω_n * x[2] + bank_sys.ω_l * x[3])), x)
    end

    return (bank_sys.γ + bank_sys.τ) - (((x[1] + x[2] + x[3]) - (bank.d + x[4])) /  (bank_sys.ω_n * x[2] + bank_sys.ω_l * x[3]))
    #return (bank_sys.γ + bank_sys.τ) - (bank_sys.banks[1].e /  (bank_sys.ω_n * x[2] + bank_sys.ω_l * x[3]))
    #return x[4] + bank.d + (bank_sys.γ + bank_sys.τ) * (bank_sys.ω_n * x[2] + bank_sys.ω_l * x[3]) - x[1] + x[2] + x[3]
end

function optim_allocation!(bank::Bank, bank_sys::BankSystem)
    
    opt              = Opt(:LD_SLSQP, 4)
    opt.lower_bounds = [0.0, 0.0, 0.0, 0.0]
    opt.xtol_rel     = 0
    opt.maxtime      = 5

    opt.min_objective = (x, fΔ) -> obj_f(x, fΔ, bank, bank_sys)
    equality_constraint!(opt,   (x, fΔ_bs) ->  bs_equality(x, fΔ_bs, bank))
    inequality_constraint!(opt, (x, fΔ_liq) -> liq_inequality(x, fΔ_liq, bank, bank_sys))
    inequality_constraint!(opt, (x, fΔ_cap) -> cap_inequality(x, fΔ_cap, bank, bank_sys))
    # profit inequality is necessery as otherwise the objective function have a domain error
    inequality_constraint!(opt, (x, fΔ_prof) -> prof_inequality(x, fΔ_prof, bank, bank_sys))

    s = bank_sys.banks[1].e + bank_sys.banks[1].d
    b = bank.d * 0
    c = bank.d * (bank_sys.α + 0.2)
    dec_params = (s + b - c)

    x0 = [c, dec_params, 0, b]

    (minf,minx,ret) = optimize(opt, x0)

    ret == :FORCED_STOP && error("NLopt status: :FORCED_STOP")
    bank.c = minx[1]
    bank.n = minx[2]
    bank.l = minx[3]
    bank.b = minx[4]
end
