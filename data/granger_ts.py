import pandas as pd
import numpy as np
import statsmodels.api as sm
import yfinance as yf
from fredapi import Fred
from scipy.linalg import lstsq as sp_lstsq
import timeit
from scipy.linalg import solve


fred = Fred(api_key='18c2830f79155831d5c485d84472811f')
region = 'eu'

if region == 'eu':
    print('region: EU')
    spread = fred.get_series('BAMLHE00EHYIOAS') # euro
    tickers = "EBO.DE RAW.DE KBC.BR CBK.DE DBK.DE NDA-SE.ST DANSKE.CO JYSK.CO SYDB.CO BBVA BKT.MC CABK.MC SAB.MC SAN.MC UNI.MC BNP.PA ACA.PA GLE.PA ALPHA.AT EUROB.AT ETE.AT TPEIR.AT OTP.BD A5G.IR BARC.L BIRG.IR BAMI.MI ISP.MI MB.MI BMPS.MI BPE.MI UCG.MI ABN.AS INGA.AS DNB.OL PKO.WA PEO.WA BCP.LS SEB-A.ST SHB-A.ST SWED-A.ST"
    banks_index = pd.read_excel("data/stoxx_banks.xlsx")\
                    .sort_index(ascending=False)\
                    .set_index('Date')
                    
    index = pd.DataFrame(yf.download("^STOXX", start="2000-01-01", group_by='tickers')['Open'])        
                    
elif region == 'us':
    print('region: US')
    spread = fred.get_series('BAMLC0A0CM') # usa
    tickers = "BAC BK BCS BMO COF SCHW C CFG DB GS JPM MTB MS NTRS PNC STT TD TFC UBS WFC ALLY AXP DFS FITB HSBC HBAN KEY MUFG PNC RF SAN"
    banks_index = pd.DataFrame(yf.download("^BKX", start="2000-01-01", group_by='tickers')['Open'])        
    index = pd.DataFrame(yf.download("^SPX", start="2000-01-01", group_by='tickers')['Open'])        


data_raw = yf.download(tickers, start="2000-01-01", group_by='tickers')

df_rets = data_raw.xs('Close', axis=1, level=1, drop_level=True)\
            .pct_change()\
            .iloc[3:, :]

# subtracting the index returns from the bank returns                
df_rets = df_rets\
            .loc[:banks_index.index[-1],:]\
            .sub(banks_index.pct_change().loc[df_rets.index[0]:,:], axis='columns', fill_value=0)\
            .iloc[:,0:-1]

granger = pd.DataFrame({'degree': []})

cor_w = 100

def na_share(df):
    return np.sum(df.isna().sum()) / (df.shape[0])

def std_err(X, y, beta):
    y = np.asarray(y)
    X = np.asarray(X)
    mse = np.mean((y - np.matmul(X, beta))**2)
    sigma_params = mse * np.linalg.inv(np.matmul(X.T, X))
    return np.sqrt(np.diag(sigma_params))

def granger_cause(df):
    pair = pd.concat([df.iloc[:, 0], df.iloc[:, 0:2].shift()], axis=1)\
        .dropna()

    # old and slow method:
    #model = sm.OLS(pair.iloc[:, 0], sm.add_constant(pair.iloc[:, 1:3]))
    #results = model.fit()
    
    pair['const'] = 1
    beta, _, _, _ = sp_lstsq(pair.iloc[:,[1,2,3]], pair.iloc[:,0], lapack_driver='gelsy', check_finite=False)
    sigma = std_err(pair.iloc[:,[1,2,3]], pair.iloc[:,0], beta)
    
    # return results.params[2], results.pvalues[2] < 0.05
    return beta[1], np.abs(beta[1] / sigma[1]) > 1.96 
    
def granger_mat(df):
    
    granger_mat = np.zeros((df.shape[1], df.shape[1]))
    
    for i in range(1, len(df.columns)):
        for j in range(1, len(df.columns)):
            
            if  i == j:
                granger_mat[i,j] = 0
                continue
            
            if (na_share(df.iloc[:,i]) > 0.95) or (na_share(df.iloc[:,j]) > 0.95):
                granger_mat[i,j] = np.nan
                continue
            
            _, signif = granger_cause(df.iloc[:, [i,j]])
            
            granger_mat[i,j] = 1 if signif else 0
            #print(df.columns[i], df.columns[j])
    return granger_mat  

for t in range(cor_w, len(df_rets)):
    print(t/len(df_rets))
    window = df_rets.iloc[(t - cor_w):t, :]
    mat = granger_mat(window)
    degree = np.nansum(mat) / (mat.shape[0] * mat.shape[1] - mat.shape[0])
    granger = pd.concat([granger, pd.DataFrame({'degree': [degree]})])

df_test = window.iloc[:, [3, 4]]

pair = pd.concat([df_test.iloc[:, 0], df_test.iloc[:, 0:2].shift()], axis=1)\
        .dropna()
            
result_sp = sp_lstsq(pair.iloc[:,[1,2,3]], pair.iloc[:,0], lapack_driver='gelsy', check_finite=False)
    
beta, _, _, _ = sp_lstsq(pair.iloc[:,[1,2,3]], pair.iloc[:,0], lapack_driver='gelsy', check_finite=False)

sigma = std_err(pair.iloc[:,[1,2,3]], pair.iloc[:,0], beta)

np.abs(beta / sigma) 

model = sm.OLS(pair.iloc[:, 0], sm.add_constant(pair.iloc[:, 1:3]))
results = model.fit()
results.summary()

model = sm.OLS(pair.iloc[:, 0], sm.add_constant(pair.iloc[:, 1:3]))
results = model.fit()

results.params[1]
results.pvalues[1] < 0.05

mse = np.mean((np.asarray(pair.iloc[:,0]) - np.asarray(np.matmul(pair.iloc[:,[1,2,3]], result_sp[0])))**2)

sigma_params = mse * np.linalg.inv(np.matmul(pair.iloc[:,[1,2,3]].T, pair.iloc[:,[1,2,3]]))

np.sqrt(np.diag(sigma_params))

std_err(pair.iloc[:,[1,2,3]], pair.iloc[:,0], result_sp[0])


      