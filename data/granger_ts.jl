@time using LinearAlgebra
@time using Statistics
@time using DataFrames
@time using CSV
@time using MarSwitching
@time using GLM
@time using BenchmarkTools
@time using Tables
@time using Plots

cor_w = 30

run(`python data/granger_ts.py
    --region eu
    --freq daily`) # daily or weekly

replace(x, to) = ismissing(x) ? to : x

data_raw = CSV.read("data/df_rets_granger.csv", DataFrame)
data = Matrix(replace.(data_raw[:, 2:end], Inf))

function na_share(df::Matrix{Float64})
    return sum(isinf.(df), dims=1) ./ size(df, 1)
end

function remove_infs(df::Matrix{Float64})
    
    infs = []
    for i in 1:size(df, 2)
        infs = [infs; findall(isinf.(df[:,i]))]
    end

    return df[(1:end) .∉ (infs,),:]
end

function std_err(X::Matrix, y::Vector, β::Vector)
    mse = mean((y .- X * β).^2)
    σ² = mse * (X'X)^(-1)
    return sqrt.(diag(σ²))
end    

function preproc_df(df::Matrix{Float64})
    pair = [df[2:end, 1] add_lags(df[:, 1], 1)[:,2] add_lags(df[:, 2], 1)[:,2]]
    pair = remove_infs(pair)
    pair = [pair ones(size(pair, 1))]
    return pair
end 


function granger_cause(df::Matrix{Float64})
    pair = preproc_df(df)

    X = pair[:, 2:end]
    y = pair[:, 1]
    
    β = (X'X) \ X'y
    Σ = std_err(X, y, β)
    
    return β[2], abs(β[2] / Σ[2]) > 1.96
end

function granger_conect(df::Matrix{Float64})
    granger_mat = zeros(size(df, 2), size(df, 2))

    for i in 1:size(df, 2)
        for j in 1:size(df, 2)

            pair = df[:, [i, j]]
            
            if i == j
                granger_mat[i, j] = 0
                continue
            end

            if any(na_share(pair) .>= 0.9)
                granger_mat[i, j] = Inf
                continue
            end

            if rank(preproc_df(pair)) != 4
                granger_mat[i, j] = Inf
                continue
            end
            
            #println(i, " ", j)
            _, signif = granger_cause(pair)

            granger_mat[i, j] = signif ? 1 : 0
        end
    end

    return granger_mat
end

replace_inf(x) = isinf(x) ? 0.0 : x

function granger_degree(x)
    return sum(replace_inf.(x)) / ((size(x)[2])^2 - size(x)[2] - sum(isinf.(x)))
end    

granger_ts = zeros(length(cor_w:size(data)[1]))

for t in cor_w:size(data)[1]
    println("$(round((t / size(data)[1])*100, digits = 2)) %")
    window = data[(t - cor_w + 1):t, :]
    mat = granger_conect(window)
    granger_ts[t - cor_w + 1] = granger_degree(mat)
end

granger_out = DataFrame(Date = data_raw.Date[cor_w:end], granger = granger_ts)

CSV.write("data/granger_ts.csv", granger_out)
