using MarSwitching
using DataFrames
using CSV
using Statistics
using Plots
using GLM

# to do:
# - odjac kurs sp500 od indeksu 
# - odjąć tez od kursu banków?

# function to remove outlier from a matrix of data


# download data
# args: region us/eu, freq weekly/daily
run(`python data/stocks_download.py --region us --freq weekly`)


function remove_outlier(data, m = 3)
    
    outliers = []
    
    for i in 1:size(data, 2)
        outliers = [outliers; findall(data[:,i] .> (mean(data[:,i]) + m * std(data[:,i])))]
    end

    return data[(1:end) .∉ (outliers,),:]
end


# Load data
data = CSV.read("data/bank_cor.csv", DataFrame)

df_model = Matrix(dropmissing(data[:, ["banks_index", "index", "cor", "spread"]]))

df_model = remove_outlier(df_model, 5)

standard(x) = (x .- mean(x)) ./ std(x)

df_model[:,1] = standard(sqrt.((df_model[:,1]).^2))
df_model[:,2] = standard(df_model[:,2] .- df_model[:,1])
df_model[:,3] = standard(df_model[:,3])
df_model[:,4] = standard(df_model[:,4])

exog = add_lags(df_model[:,1], 1)[:,2]
exog_switch = add_lags(df_model[:,3],1)[:,2] #[df_model[2:end, 3] df_model[2:end,2]]

tvtp = [ones(length(exog[:,1])) df_model[2:end,2]]

model = MSModel(df_model[2:end,1], 2, 
                exog_vars = exog,
                exog_switching_vars = exog_switch,
                exog_tvtp = tvtp,
                random_search_em = 20,
                random_search = 3
                )

summary_msm(model)

any(isnan.(tvtp))
cor(df_model[:,1], df_model[:,2])

plot(df_model[:,1])

ed = expected_duration(model)[expected_duration(model)[:,2] .< 200,:]
plot(ed, label = ["Calm market conditions" "Volatile market conditions"],
         title = "Time-varying expected duration of regimes") 

mean(expected_duration(model), dims = 1)

plot(smoothed_probs(model),
         label     = ["Calm market conditions" "Volatile market conditions"],
         title     = "Regime probabilities", 
         linewidth = 2,
         legend = :bottomleft)

df_ols = DataFrame(df_model, :auto)[2:end, :]
df_ols[!, "lag"] = exog
df_ols[!, "lag_cor"] = exog_switch
ols = lm(@formula(x1 ~ lag + lag_cor), df_ols)

