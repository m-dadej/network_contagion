@time using LinearAlgebra
@time using Statistics
@time using DataFrames
@time using CSV
@time using MarSwitching
@time using GLM

replace(x, to) = ismissing(x) ? to : x

data = CSV.read("data/df_rets_granger.csv", DataFrame)[:, 2:end]
data = Matrix(replace.(data, Inf))



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

function std_err(X, y, β)
    mse = mean((y .- X * β).^2)
    σ² = mse * (X'X)^(-1)
    return sqrt.(diag(σ²))
end    

add_lags(data[900:1000, 26], 1)

function granger_cause(df::Matrix{Float64})
    
    pair = [df[2:end, 1] add_lags(df[:, 1], 1)[:,2] add_lags(df[:, 2], 1)[:,2]]
    pair = remove_infs(pair)
    pair = [pair ones(size(pair, 1))]

    β = (pair[:,2:end]'*pair[:,2:end])\pair[:,2:end]'*pair[:,1]
    Σ = std_err(pair[:,2:end], pair[:,1], β)
    
    return β[2], abs(β[2] / Σ[2]) > 1.96
end

function granger_conect(df::Matrix{Float64})
    granger_mat = zeros(size(df, 2), size(df, 2))

    for i in 1:size(df, 2)
        for j in 1:size(df, 2)
            if i == j
                granger_mat[i, j] = 0
                continue
            end

            if any(na_share(df[:, [i, j]]) .>= 0.9)
                granger_mat[i, j] = Inf
                continue
            end
            
            #println(i, " ", j)
            _, signif = granger_cause(df[:, [i, j]])

            granger_mat[i, j] = signif ? 1 : 0
        end
    end

    return granger_mat
end

window[:, [16, 29]]
data[(t - cor_w + 1):t, :][:, [16, 29]]

replace_inf(x) = isinf(x) ? 0.0 : x

function granger_degree(x)
    return sum(replace_inf.(x)) / ((size(x)[2])^2 - size(x)[2] - sum(isinf.(x)))
end    

granger_ts = zeros(length(cor_w:size(df)[1]))

cor_w = 100

for t in cor_w:size(data)[1]
    println(t / size(data)[1])
    window = data[(t - cor_w + 1):t, :]
    mat = granger_conect(window)
    granger_ts[t - cor_w + 1] = granger_degree(mat)
end

df[(t - cor_w + 1):t, [16, 29]]
window[:, [16, 29]]

df = data[(t - cor_w + 1):t, :]
df = window
granger_cause(df[:, [15, 28]])
window
granger_conect(df[(t - cor_w + 1):t, :])
window[:, [16, 29]]



df_t = df[:, [15, 28]]

remove_infs(df[:, [15, 28]])

pair = [df_t[2:end, 1] add_lags(df_t[:, 1], 1)[:,2] add_lags(df_t[:, 2], 1)[:,2]]
pair = remove_infs(pair)
pair = [pair ones(size(pair, 1))]

β = (pair[:,2:end]'*pair[:,2:end])\pair[:,2:end]'*pair[:,1]
Σ = std_err(pair[:,2:end], pair[:,1], β)

return β[2], abs(β[2] / Σ[2]) > 1.96


    