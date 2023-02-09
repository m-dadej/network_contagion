using Random
using Distributions
using Plots
using LinearAlgebra


function sum_balance(A_ib, class)
    n = size(A_ib)[1]
    if class == "assets"
        return [sum(A_ib[i, 1:n]) for i in 1:n]
    else
        return [sum(A_ib[1:n, i]) for i in 1:n]
    end    
end

function is_solvent(bank, A_ib, D, A_M, ϕ, q)
    (1-ϕ) .* sum_balance(A_ib, "assets")[bank] .+ q .* A_M[bank] .- sum_balance(A_ib, "liab")[bank] - D[bank].> 0
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

function funds_matching(Δ)
    
    n = size(Δ)[1]

    fund_matching = Model(HiGHS.Optimizer)
    @variable(fund_matching, A_ib[1:n, 1:n] >= 0)

    # borrowers constraint
    for i in 1:n 
        @constraint(fund_matching, sum(A_ib[:,i]) <= abs.(clamp(Δ[i], -Inf, 0)))
    end
    
    # savers constraint
    for i in 1:n 
        @constraint(fund_matching, sum(A_ib[i,:]) == clamp(Δ[i], 0, Inf))
    end

    @objective(fund_matching, Min, sum(A_ib[i,:]'A_ib[i,:] for i in 1:n))
    optimize!(fund_matching)

    return value.(A_ib)
end

n = 10
init_b = 0.01
exp_A_M = 90
σ_A_M = 15
exp_buffer_rate = 0.04
σ_buffer_rate = 0.005
exp_A_ib = 20
exp_D = 80
σ_D = 15

# A_M + A_ib + cash = D + I + e
# gdzie e > 0.04 * (A_M + A_ib)

A_M = rand(Normal(exp_A_M,σ_A_M), n)
A_ib = zeros(n,n)
buffer_rate = rand(Normal(exp_buffer_rate, σ_buffer_rate), n)
D = rand(Normal(exp_D, σ_D), n)
Δ = D .- A_M .+ (D .* buffer_rate) 
k = (buffer_rate) .* D

# (A_M .+ Δ) .- (D .+ k) # balance sheet equality
# k ./ (D .+ k) # actual capital ratio
sum(Δ)
A_ib = funds_matching(Δ)

sum_balance(A_ib, "assets")

D = D .+ clamp.(Δ, 0, Inf) .- sum_balance(A_ib, "assets")
A_M = A_M .- abs.(clamp.(Δ, -Inf, 0)) .- sum_balance(A_ib, "liab")

(A_M .+ sum_balance(A_ib, "assets")) .- (D .+ k .+ sum_balance(A_ib, "liab"))

sum_balance(A_ib, "assets") .- clamp.(Δ, 0, Inf)

# workspace ------------
findall(x -> x .< 0, Δ)

rand([1,0], 4,4)

n = 4
l = rand([1,0], n,n) .* rand(Normal(10,3), n)

l * repeat([1],n)


penalty = reduce(vcat,transpose.([[10^2, 0, 10^2, 0] for _ in 1:n]))

l' + penalty 

w1 = [4,0,0,0]

w1 = [2,0,0,2]

w1 = [1,1,1,1]

w =  reduce(vcat,transpose.([w1 for _ in 1:4]))
sum(w1.^2)
w1'w1

ones(4)'w'w * ones(4)

sum(w[i,:]'w[i,:] for i in 1:4)


w1'w1

[1,2,3,4]'*[1,0,1,0]

sum([1,2,3,4] .* [1,0,1,0])

w[1,:]

sum(w[:,1] .* Δ)

n = 4
model = Model(HiGHS.Optimizer)
Δ = sample(collect(1:10), n)
Δ[isodd.(collect(1:n))] .= 0
γ = repeat([1], n)
γ[isodd.(collect(1:n))] .= 0




@variable(model, l[1:n, 1:n] >= 0)

for i in 1:n 
    @constraint(model, sum(l[:,i] * γ[i]) == Δ[i])
end

@objective(
    model,
    Min,
    sum(l[i,:]'l[i,:] for i in 1:n)
)

print(model)

optimize!(model)

objective_value(model)

value.(l)

sum(w[:,1] * γ[2]) == Δ[2]
