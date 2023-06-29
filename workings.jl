

include("risk_heterogeneity.jl")

sigma_pars = 0.4:0.2:2.5
sigma_pars = collect(sigma_pars) .+ 0.01
r_l_pars = 0.02:0.01:0.12
param_mat = zeros(length(r_l_pars), length(sigma_pars))

i = 0
j = 0
bank = 1

for sigma in sigma_pars 
    j += 1    
    for r_l in r_l_pars
        i += 1
        bs = optim_allocation(d[bank], α, ω_n, ω_l, γ, τ, e[bank], r_n[bank] , r_l, ζ, exp_δ, σ_rn, σ_δ, sigma)
        param_mat[i, j] = round((bs[3] / (d[bank] + e[bank] + bs[4])) * 100)
        println("sigma: ", sigma, " | r_l: ", r_l)
    end
    i = 0
end

balance_check(bs[1], bs[2], bs[3], d[bank], bs[4], e[bank])
#round(min(bs[3], bs[4])) # round((bs[1] / d[bank]) * 100)
#round(equity_requirement(bs[1], bs[2], bs[3], d[bank], bs[4], ω_n, ω_l)*100)
optim_allocation(d[bank] , α, ω_n, ω_l, γ, τ, e[bank], r_n[bank]-0.05 , 0.1, ζ, exp_δ, σ_rn, σ_δ, 2.01)

Random.seed!(33)
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
ζ     = 0.6 # lgd
exp_δ = 0.01 # pd
σ_δ   = 0.006 # variance of pd
r_n   = rand(Uniform(0.01, 0.15), N) # return on non liquid assets
σ_rn  = (1/12).*(maximum(r_n) - minimum(r_n)).^2


@time param = equilibrium(N, d, e, α, ω_n, ω_l, γ, τ, σ, ζ, exp_δ, σ_δ, r_n, σ_rn, N*2)

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

# l == b
sum(optim_vars[:, 3]) - sum(optim_vars[:, 4])

k = @. equity_requirement(optim_vars[:, 1], optim_vars[:, 2], optim_vars[:, 3], d, optim_vars[:, 4], ω_n, ω_l)
A = optim_vars[:,1] .+ optim_vars[:,2] .+ optim_vars[:,3]

optim_vars[:, 3] ./ A

mean(optim_vars[:, 4] ./ A)

optim_vars

A_ib = fund_matching(optim_vars[:, 3], optim_vars[:, 4], σ, e, A, 0.25)

optim_vars

l_ = optim_vars[:,3]
b_ = optim_vars[:,4]

N = size(l_)[1]
fund_matching_optim = Model(HiGHS.Optimizer)
set_silent(fund_matching_optim)
@variable(fund_matching_optim, A_ib[1:N, 1:N] >= 0)    

# borrowers constraint
for i in 1:N
    @constraint(fund_matching_optim, sum(A_ib[:,i]) == b_[i])
end
    
# savers constraint
for i in 1:N 
    @constraint(fund_matching_optim, sum(A_ib[i,:]) == l_[i])
end

# no self trading
for i in 1:N
    @constraint(fund_matching_optim, A_ib[i,i] == 0)
end

for i in 1:N 
    @constraint(fund_matching_optim, A_ib[:, i] ./ A[i] .<= 0.2)  
end

k_ = (e ./ A)

@objective(fund_matching_optim, Max,  sum(σ[i] * (A_ib[i,:]'k_) for i in 1:N))
JuMP.optimize!(fund_matching_optim)
k


A_ib = fund_matching(optim_vars[:, 3], optim_vars[:, 4], σ, k, A, 0.2)

any(-0.001 .< balance_check(optim_vars[:,1], optim_vars[:,2], sum(A_ib, dims=2), d, sum(A_ib, dims=1)', e) .> 0.001)

sum(sum(A_ib, dims=2) .- optim_vars[:, 3])
sum(sum(A_ib, dims=1) .- optim_vars[:, 4])

sum(clearing_vector(A_ib, optim_vars[:,1], optim_vars[:,2]) .- sum(A_ib, dims=1)' .< 0)

optim_vars[:,1] ./ d

print("liq: ",mean(optim_vars[:,1] ./ d), " | IB/e : ", mean(optim_vars[:, 5] ./ e))

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

sum(e .<= 0)
optim_vars

balance_check(optim_vars[:,1], optim_vars[:,2], sum(A_ib, dims=2), d, sum(A_ib, dims=1)', e)

sum(clearing_vector(A_ib, optim_vars[:,1], optim_vars[:,2]) .- sum(A_ib, dims=1)' .< 0)

max.(clearing_vector(A_ib, optim_vars[:,1] .- debt_call', optim_vars[:,2]) .- sum(A_ib, dims=1)', -sum(A_ib, dims=2))

(A_ib[calling_banks, :] .* -inf_zero(debt_call[call_id] ./ A_ib[call_id, :]))

inf_zero(x) = ifelse.(isinf.(x), 0, x)



A_ib[calling_banks, :] ./ sum(A_ib[calling_banks, :]) .* -debt_call[calling_banks]

clearing_vector(A_ib, optim_vars[:,1] .- (A_ib[call_id, :] ./ sum(A_ib[call_id, :]) .* -debt_call[call_id])', optim_vars[:,2]) .- sum(A_ib, dims=1)'


debt_call = (A_ib[call_id, :] .* -inf_zero(debt_call[call_id] ./ A_ib[call_id, :]))



[0,0,3,0]

findall

[2,0,0,1]


clearing_vector(A_ib, optim_vars[:,1], [0 for _ in 1:20]) .- sum(A_ib, dims=1)'

clearing_vector(A_ib, optim_vars[:,1], optim_vars[:,2]) .- sum(A_ib, dims=1)'
clearing_vector(A_ib, optim_vars[:,1], [0 for _ in 1:N]) .- sum(A_ib, dims=1)'

optim_vars[new_defaults,:]
clearing_vector(A_ib, optim_vars[:,1], optim_vars[:,2])[new_defaults]
sum(A_ib, dims=1)'[new_defaults]

new_defaults = rand(1:N,1)
e[new_defaults] .= -10
optim_vars[new_defaults, 1] .= 0
optim_vars[new_defaults, 2] .= 0
defaults_t = [0,1]
defaults_set = []



while defaults_t[end] != defaults_t[end-1]
    new_defaults = findall(e .<= 0)[1][findall(e .<= 0)[1] .∉ Ref(defaults_set)] # new banks with e <= 0
    A_ib[:, new_defaults] .= A_ib[:, new_defaults] .* ζ # credit loss
    balance_check(optim_vars[:,1], optim_vars[:,2], sum(A_ib, dims=2), d, sum(A_ib, dims=1)', e)
end




i = 1
optim_vars[i,1] + optim_vars[i,2] + sum(p .* get_asset_share(A_ib, i))
p̄[i]


termination_status(market_clearing)

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



get_asset_share(A_ib, 3)

sum(A_ib[:,1])

A_ib[]


ifelse.(isnan.([A_ib[1, k] / sum(A_ib[:,k]) for k in 1:N]), 0, ([A_ib[1, k] / sum(A_ib[:,k]) for k in 1:N]))

([A_ib[1, k] / sum(A_ib[:,k]) for k in 1:N])

[A_ib[k,i] / sum(A_ib[:,i]) for k in 1:20 for i in 1:20]

[sum(A_ib[:,i]) for i in 1:20]

[sum(A_ib[k, :] ./ [sum(A_ib[:,i]) for i in 1:20]) for k in 1:20]
 
[sum(A_ib[:,i]) for i in 1:20] # total liabs

# initial default
# pętla:
#   - identyfikacja nowych bankrutów
#   - credit loss wierzycieli nowych bankrutów
#   - 

# JAk bank upada to co sie dzieje z aktywami? - sa rozdysponowane pomiedzy wierzycieli?

# czy po pierwszej fali bankructw banki moga znow robic rebalancing?
#   powinny bo na tym ma m.in polegac endogenous networks
#   nie wiem z drugiej strony jak to wprowadzic

# moj kod eisenberg noe

# market clearing condition
# c + n + suma_k( η_k * l_k,i) = d + suma_k(ϵ_i b_i,k)
# cash + external assets + sum of loans avaiable to repay = deposits + sum of possible amount of liabilities to repay

# net worth 
#  w = c + sum_k(η_k * l_k,i) - d - sum_k(b_i,k)

# share of loan bank i may repay
# ϵ_i = (c + sum_k(η * p_k,i) - d)

# η_i = 1 if ϵ_i > 1 and
# η_i = 0 if ϵ_i < 0

# i.e. bank gets his asset repaid if he can repay his loan

# η_i = Θ(η_k,k=\=i) = min(1, max((c + sum_k(η * p_k,i) - d), 0))

# Θ() is a function that maps variable into 0 or 1

function θ(η)
    return min.(1, max.(η, 0))
end

# sum(A_ib, dims=2) - loans

# (c + n + l) - (d + b + e)
# e = l + c + n - d - b
e .- ((sum(A_ib, dims=2) .+ optim_vars[:,1] .+ optim_vars[:,2]) .- (d .+ sum(A_ib, dims=1)'))



# eisenberg noe --------

# L = A_ib
# A = L'
# Ae = optim_vars[:,2];
# Le = d;

epsilon = 10^(-5)
max_counts = 10^5

nbanks = size(A_ib)[1]

l = sum(A_ib, dims=1)'

# equityZero = Ae - Le + sum(A,2) - l;
equity_en = deepcopy(e)

error = 1
counts = 1

while (error > epsilon)&&(counts < max_counts)
    
    oldEquity = deepcopy(equity_en)
    recoveryVector = ones(nbanks).*((equityZero .>= 0) + (maximum(1 .+ equityZero ./ l, 0) ).*(equityZero .< 0));
    equity = Ae - Le + A*(recoveryVector) - l; 
     
    error = norm(equity - oldEquity)/norm(equity);
    counts = counts +1;
end

ones(nbanks) .* ((equity_en .>= 0) + (maximum(1 .+ equity_en ./ l, 0)))

recoveryVector = ones([nbanks 1]).*((equity>=0) + (max(1 + equity./l,0) ).*(equity<0));

equity = Ae - Le + A*(recoveryVector) - l;

paymentVector = max(0,min( Ae - Le + A*recoveryVector, l)); 

equityLoss = equity - equityZero;

## 

optim_vars[:, 1] ./ d # liquidity ratio
optim_vars[:, 5] ./ e # roe
k = @. equity_requirement(optim_vars[:, 1], optim_vars[:, 2], optim_vars[:, 3], d, optim_vars[:, 4], ω_n, ω_l)
optim_vars[:, 5] ./ (optim_vars[:,1] .+ optim_vars[:, 2] .+ optim_vars[:, 3]) 
(optim_vars[:, 2] .+ optim_vars[:, 3])

mean((optim_vars[:, 2] .+ optim_vars[:, 3] .+ optim_vars[:, 1]) ./ e)

optim_vars[:, 3] ./ (optim_vars[:, 1] .+ optim_vars[:, 2] .+ optim_vars[:, 3])

mean(optim_vars[:, 1] ./ d)
mean(optim_vars[:, 5] ./ e)
mean(@. equity_requirement(optim_vars[:, 1], optim_vars[:, 2], optim_vars[:, 3], d, optim_vars[:, 4], ω_n, ω_l))
mean(optim_vars[:, 5] ./ (optim_vars[:,1] .+ optim_vars[:, 2] .+ optim_vars[:, 3]))



