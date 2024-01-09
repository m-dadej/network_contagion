using Revise
using MarSwitching
using DataFrames
using CSV
using Statistics
using Plots

#using Pkg; Pkg.add.(["Distributions", "MarSwitching", "DataFrames", "CSV", "Statistics", "Plots", "GLM"])
# to do:
# - odjac kurs sp500 od indeksu 
# - odjąć tez od kursu banków?

# odejmij jakis AR proces of R


# download data
# args: region us/eu, freq weekly/daily
run(`python data/stocks_download.py
    --region eu
    --freq daily
    --cor_window 63
    --eig_k 5
    --excess False`)

function remove_outlier(data, m = 3)
    
    outliers = []
    for i in 1:size(data, 2)
        outliers = [outliers; findall(data[:,i] .> (mean(data[:,i]) + m * std(data[:,i])))]
    end

    return data[(1:end) .∉ (outliers,),:]
end


# Load data
data = CSV.read("C:/Users/HP/Documents/julia/finansowe/contagion/data/bank_cor.csv", DataFrame)
#data = CSV.read("data/bank_cor.csv", DataFrame)
#granger_df = CSV.read("data/granger_ts.csv", DataFrame)
granger_df = CSV.read("C:/Users/HP/Documents/julia/finansowe/contagion/data/granger_ts.csv", DataFrame)

data = sort(leftjoin(data, granger_df, on = :Date), :Date)

df_model = Matrix(dropmissing(data[:, ["banks_index", "index", "spread", "eig"]]))

df_model = remove_outlier(df_model, 5)

standard(x) = (x .- mean(x)) ./ std(x)
ols(y, X) = (X'X) \ X'y

X_mean = [ones(length(df_model[2:end,1])) add_lags(df_model[:,2], 1)]
β_mean = ols(df_model[2:end,1] ,X_mean)
y_hat = df_model[2:end,1] .- X_mean*β_mean 

# y_hat = sqrt.((df_model[:,1] .- mean(df_model[:,1])).^2)
# df_model[:,2] = sqrt.(df_model[:,2].^2)

# for col in 1:4
#     df_model[:,col] = standard(df_model[:,col])
# end

exog = add_lags(abs.(y_hat), 1)[:,2]
#exog = add_lags(df_model[:,1], 1)[:,2]
exog_switch = add_lags(df_model[2:end,4],1)[:,2] #[df_model[2:end, 3] df_model[2:end,2]]

tvtp = [ones(length(exog[:,1])) add_lags(df_model[:,3], 1)[2:end,2]]
tvtp[:, 2] = standard(tvtp[:, 2])


model = MSGARCH(abs.(y_hat[2:end]), 2, 
                exog_vars = exog,
                exog_switching_vars = exog_switch,
                # exog_tvtp = tvtp,
                # random_search_em = 10,
                random_search = 3
                )

summary_msm(model)

plot(sqrt.((df_model[:,1] .- df_model[:,2]).^2))
plot(sqrt.((df_model[:,1]).^2))
plot(data.granger)

cor(Matrix(dropmissing(data[:, ["cor", "eig"]])))

plot(Matrix(dropmissing(data[:, ["cor", "eig"]])))

cor(df_model[2:end,1], exog_switch)

ed = expected_duration(model)
plot(ed, label = ["Calm market conditions" "Volatile market conditions"],
         title = "Time-varying expected duration of regimes") 

mean(expected_duration(model), dims = 1)

plot(smoothed_probs(model)[end-500:end,:],
         label     = ["Calm market conditions" "Volatile market conditions"],
         title     = "Regime probabilities", 
         linewidth = 0.5,
         legend = :bottomleft)

df_ols = DataFrame(df_model, :auto)[2:end, :]
df_ols[!, "lag"] = exog
df_ols[!, "lag_cor"] = exog_switch
ols = lm(@formula(x1 ~ lag + lag_cor), df_ols)

