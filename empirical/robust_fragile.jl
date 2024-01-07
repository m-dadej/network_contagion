using MarSwitching
using DataFrames
using CSV
using Statistics
using Plots
using GLM

#using Pkg; Pkg.add.(["Distributions", "MarSwitching", "DataFrames", "CSV", "Statistics", "Plots", "GLM"])
# to do:
# - odjac kurs sp500 od indeksu 
# - odjąć tez od kursu banków?


# download data
# args: region us/eu, freq weekly/daily
run(`python data/stocks_download.py
    --region eu
    --freq weekly
    --cor_window 100
    --eig_k 2
    --excess True`)

function remove_outlier(data, m = 3)
    
    outliers = []
    for i in 1:size(data, 2)
        outliers = [outliers; findall(data[:,i] .> (mean(data[:,i]) + m * std(data[:,i])))]
    end

    return data[(1:end) .∉ (outliers,),:]
end


# Load data
data = CSV.read("data/bank_cor.csv", DataFrame)
# granger_df = CSV.read("data/granger_ts.csv", DataFrame)

# data = leftjoin(data, granger_df, on = :Date)

df_model = Matrix(dropmissing(data[:, ["banks_index", "index", "spread", "cor_lw"]]))

df_model = remove_outlier(df_model, 5)

standard(x) = (x .- mean(x)) ./ std(x)

df_model[:,1] = standard(sqrt.((df_model[:,1]).^2))
df_model[:,2] = standard(df_model[:,2])
df_model[:,3] = standard(df_model[:,3])
df_model[:,4] = standard(df_model[:,4])

exog = [add_lags(df_model[:,1], 1)[:,2] df_model[2:end,2]]
exog_switch = add_lags(df_model[:,4],1)[:,2] #[df_model[2:end, 3] df_model[2:end,2]]

tvtp = [ones(length(exog[:,1])) add_lags(df_model[:,3], 1)[:,2]]

# df_model[:,1] = standard(sqrt.(df_model[:,1].^2))
# df_model[:,2] = standard(sqrt.((df_model[:,2] .- df_model[:,1]).^2))
# df_model[:,3] = standard(df_model[:,3])
# df_model[:,4] = standard(df_model[:,4])

# exog = add_lags(df_model[:,1], 1)[:,2]
# exog_switch = add_lags(df_model[:,4],1)[:,2]
# tvtp = [ones(length(exog[:,1])) add_lags(df_model[:,3],1)[:,2]]

model = MSModel(df_model[2:end,1], 2, 
                exog_vars = exog,
                exog_switching_vars = exog_switch,
                exog_tvtp = tvtp,
                random_search_em = 10
                )

summary_msm(model)

plot(sqrt.((df_model[:,1] .- df_model[:,2]).^2))
plot(sqrt.((df_model[:,1]).^2))
plot(data.cor_lw)

cor(Matrix(dropmissing(data[:, ["cor", "eig"]])))

plot(Matrix(dropmissing(data[:, ["cor", "eig"]])))

cor(df_model[2:end,1], exog_switch)

ed = remove_outlier(expected_duration(model), 1)
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

