using Random
using Distributions
using Plots
using LinearAlgebra

# - w funkcji  contagion(): (sum_balance(A_ib, "assets")[1:n] to chyba nie jest potrzebne? Zawsze bedzie tyle samo
# - czy po initial bancruptcy ten wylosowany bank moze wciaz życ? W koncu jedynie usuwamy mu liabilities?
# potem chyba tez - usuwa sie im liab, zostaja A_M i nagle maja dodatni capital
# - chyba trzeba zrobic tak ze pozostale aktywa po defaulcie sa dzielone na wierzycieli

function make_asset_graph(n, tot_assets, integ)
    
    # alternative: A_ib = reduce(hcat,[shuffle(vcat([1 for _ in 1:z], [0 for _ in 1:(n-z)])) for _ in 1:n])'
    A_ib = rand(Bernoulli(integ), n,n)

    for i in 1:n
        A_ib[i,i] = 0
    end

    # if no link then 1 random
    for bank in findall([sum(A_ib[i, 1:n]) != 0 for i in 1:n])
        possible_links = [x for x in 1:n if x != bank]
        A_ib[bank, sample(possible_links, 1)] .= 1
    end 
    A_ib = tot_assets ./ [sum(A_ib[i, 1:n]) for i in 1:n] .* A_ib

    return A_ib
end

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

function contagion(n, init_b, exp_A_M, exp_A_ib, buffer_rate, integ)

    A_M = rand(Normal(exp_A_M,1), n) # illiquid exogenous assets (e.g. mortgage)
    A_ib = make_asset_graph(n, exp_A_ib, integ) # interbank loans from bank i to j

    D = (sum_balance(A_ib, "assets")[1:n] .+ A_M[1:n] .* (1 - buffer_rate)) .- sum_balance(A_ib, "liab")[1:n] # deposits top-up so that capital buffer is 4%
    A_M[sample(1:n, init_b)] .= 0

    n_default = [0]
    n_default = push!(n_default, sum(capital_buffer(collect(1:n), A_ib, D, A_M) .< 0))

    while n_default[end] != n_default[end-1]
        A_ib[1:n, capital_buffer(collect(1:n), A_ib, D, A_M) .< 0] .= 0
        push!(n_default, sum(capital_buffer(collect(1:n), A_ib, D, A_M) .< 0))
    end  

    return n_default
end

function contagion2(n, init_b, exp_A_M, exp_A_ib, buffer_rate, integ)

    A_M = rand(Normal(exp_A_M,0), n) # illiquid exogenous assets (e.g. mortgage)
    A_ib = make_asset_graph(n, exp_A_ib, integ) # interbank loans from bank i to j
    
    D = (sum_balance(A_ib, "assets")[1:n] .+ A_M[1:n] .* (1 - buffer_rate)) .- sum_balance(A_ib, "liab")[1:n] # deposits top-up so that capital buffer is 4%
    A_M[sample(1:n, init_b)] .= 0
    
    defaults_t = [0]
    defaults = []
    defaults = unique(vcat(defaults, findall(capital_buffer(collect(1:n), A_ib, D, A_M) .< 0)))
    push!(defaults_t, length(defaults))
    
    while defaults_t[end] != defaults_t[end-1]
        A_ib[1:n, capital_buffer(collect(1:n), A_ib, D, A_M) .< 0] .= 0
        defaults = unique(vcat(defaults, findall(capital_buffer(collect(1:n), A_ib, D, A_M) .< 0)))
        push!(defaults_t, length(defaults))
    end  

    return defaults_t
end

n = 1000
sys_threshold = 0.05
n_runs = 100
results = []
Random.seed!(1234)

for z in 0:0.0005:0.011
    runs = []
    for contagion in 1:n_runs
        push!(runs, contagion2(n, 1, 80, 20, 0.04, z)[end])     
    end
    systematic = filter(x -> x > n * sys_threshold, runs)
    systematic = length(systematic) == 0 ? 0 : mean(systematic)
    push!(results, [z, systematic, sum(runs .> n * sys_threshold) / n_runs])
    println("calculating contagion with z = ", z, " | extent: ", systematic, " | default prob: ", sum(runs .> n * sys_threshold) / n_runs)
end

results = (reduce(hcat, results[1:end])')
results[:, 2] = results[:, 2] ./ 1000

contagion_plot = plot(results[:,1], results[:,2:3],  label=["Extent of contagion" "Frequency of contagion"])
plot!(legend=:outerbottom, legendcolumns=2)
savefig(contagion_plot, "research_proposal/contagionplot.png")


