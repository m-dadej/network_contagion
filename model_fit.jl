
include("risk_heterogeneity.jl")

N     = 20 # n banks
α     = 0.1 # liquidity requirement
ω_n   = 1 # risk weight on non-liquid assets
ω_l   = 0.2 # risk weight on liquid assets
γ     = 0.08 # equity requirement ratio
τ     = 0.01 # equity buffer
d = [ 312.86,   71.54,  184.45,  682.65,  429.67,  192.09,  142.42, 105.5 ,  142.97,  232.6 ,  176.82,  419.89,  994.65,   53.56, 43.49,   63.92,   96.45, 1477.45,  672.12,   67.27]
e = [19.67,   6.91,  21.73,  38.5 , 112.64,  11.8 ,   8.87,  15.55, 212.8 ,  61.92, 135.13,  30.43, 129.53,   9.21,  16.44,  12.98, 12.2 , 312.26,  97.85,  15.05]
#d     = [(606/1.06), (807/1.5), (529/1.08), (211/0.7), (838/1.47), (296/0.63), (250/0.68), (428/2), (284/1.24), (40/0.94), (8.2/0.2), (252/1.74), (24/0.19), (111.1/1.03), (88.9/1.3), (51.8/0.42), (63/0.48), (111.1/1.65), (100/1.37), (11.6/0.15)] # rand(Normal(700, 100), N) # deposits
#e     = [55.6, 90.0, 48.5, 53.0, 81.0, 53.0, 57.0, 48.0, 26.0, 43.0, 20.0, 23.0, 16.0, 10.0, 8.0, 5.0, 6.0, 10.0, 9.0, 9.0] #rand(Normal(50, 20), N) # equity
#d     = rand(Normal(500, 50), N)
#e     = rand(Normal(40, 5), N)
σ     = rand(Normal(3, 0), N) .+ 0.0001 # risk aversion
#σ[rand(1:N, 1)] .= -2
ζ     = 0.7 # lgd
exp_δ = 0.03 # pd
σ_δ   = 0.006 # variance of pd
r_n   = rand(Uniform(0.01, 0.15), N) # return on non liquid assets
σ_rn  = (1/12).*(maximum(r_n) - minimum(r_n)).^2

function model_fit(N, d, e, α, ω_n, ω_l, γ, τ, σ, ζ, exp_δ, σ_δ, r_n, σ_rn)

    param = equilibrium(N, d, e, α, ω_n, ω_l, γ, τ, σ, ζ, exp_δ, σ_δ, r_n, σ_rn, N*2)

    eq_r_l = param.r_l[findmin(abs.(param.imbalance))[2]-1]
    
    minimum(abs.(param.imbalance))
    
    optim_vars = zeros(N, 5)
    
    # c, n, l, b
    for bank in 1:N 
        println("liczenie #", bank, " banku")
        optim_vars[bank, 1:4] .= optim_allocation(d[bank], α, ω_n, ω_l, γ, τ, e[bank], r_n[bank], eq_r_l, ζ, exp_δ, σ_rn, σ_δ, σ[bank])
        optim_vars[bank, 5] = profit(r_n[bank], optim_vars[bank, 2], eq_r_l, optim_vars[bank, 3], ζ, exp_δ, optim_vars[bank, 4], optim_vars[bank, 5])
    end
    
    length(findall(-0.001 .< balance_check(optim_vars[:,1], optim_vars[:,2], optim_vars[:, 3], d, optim_vars[:,4], e) .> 0.001)) > 0 && error("non-feasible")
    
    # imbalance adjustment 
    imbalance = sum(optim_vars[:,3]) - sum(optim_vars[:,4])
    
    if imbalance > 0
        borrowers = findall(optim_vars[:,4] .> 0.0001)
        optim_vars[borrowers, 4] .+= imbalance ./ length(borrowers)
        optim_vars[borrowers, 1] .+= imbalance ./ length(borrowers)
    elseif imbalance < 0
        lenders = findall(optim_vars[:, 3] .> 0.0001)
        optim_vars[lenders, 3] .-= imbalance ./ length(lenders)
        d[lenders] .-= imbalance ./ length(lenders)
    end    

    return optim_vars
end

res = model_fit(N, d, e, α, ω_n, ω_l, γ, τ, σ, ζ, exp_δ, σ_δ, r_n, σ_rn)

res = hcat(res, res[:,1] .+ res[:, 2] .+ res[:, 3])


mean((res[:, 6] .- res[:, 4] .- d) ./ res[:, 3])



mean(res[:, 2] ./ res[:, 6])

mean(res[:, 3] ./ res[:, 6])

n_sim = 7
fit = zeros(n_sim, 3)

for sim in 1:n_sim
        
    N     = 20 # n banks
    α     = 0.01 # liquidity requirement
    ω_n   = 1 # risk weight on non-liquid assets
    ω_l   = 0.2 # risk weight on liquid assets
    γ     = 0.08 # equity requirement ratio
    τ     = 0.01 # equity buffer
    d = [ 312.86,   71.54,  184.45,  682.65,  429.67,  192.09,  142.42, 105.5 ,  142.97,  232.6 ,  176.82,  419.89,  994.65,   53.56, 43.49,   63.92,   96.45, 1477.45,  672.12,   67.27]
    e = [19.67,   6.91,  21.73,  38.5 , 112.64,  11.8 ,   8.87,  15.55, 212.8 ,  61.92, 135.13,  30.43, 129.53,   9.21,  16.44,  12.98, 12.2 , 312.26,  97.85,  15.05]
    #d     = [(606/1.06), (807/1.5), (529/1.08), (211/0.7), (838/1.47), (296/0.63), (250/0.68), (428/2), (284/1.24), (40/0.94), (8.2/0.2), (252/1.74), (24/0.19), (111.1/1.03), (88.9/1.3), (51.8/0.42), (63/0.48), (111.1/1.65), (100/1.37), (11.6/0.15)] # rand(Normal(700, 100), N) # deposits
    #e     = [55.6, 90.0, 48.5, 53.0, 81.0, 53.0, 57.0, 48.0, 26.0, 43.0, 20.0, 23.0, 16.0, 10.0, 8.0, 5.0, 6.0, 10.0, 9.0, 9.0] #rand(Normal(50, 20), N) # equity
    #d     = rand(Normal(500, 50), N)
    #e     = rand(Normal(40, 5), N)
    σ     = rand(Normal(2, 0), N) .+ 0.0001 # risk aversion
    #σ[rand(1:N, 1)] .= -2
    ζ     = 0.6 # lgd
    exp_δ = 0.01 # pd
    σ_δ   = 0.006 # variance of pd
    r_n   = rand(Uniform(0.01, 0.1), N) # return on non liquid assets
    σ_rn  = (1/12).*(maximum(r_n) - minimum(r_n)).^2

    res = model_fit(N, d, e, α, ω_n, ω_l, γ, τ, σ, ζ, exp_δ, σ_δ, r_n, σ_rn)
    res = hcat(res, res[:,1] .+ res[:, 2] .+ res[:, 3])
    fit[sim, 1] = mean(res[:, 1] ./ res[:, 6])
    fit[sim, 2] = mean(res[:, 2] ./ res[:, 6])
    fit[sim, 3] = mean(res[:, 3] ./ res[:, 6])

    print((mean(r_n), σ_rn))
end

mean(fit, dims = 1)