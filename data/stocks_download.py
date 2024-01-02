import pandas as pd
import yfinance as yf
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from fredapi import Fred
from typing import Dict
import argparse
from numpy.linalg import eig
import statsmodels.api as sm


# Create the parser
parser = argparse.ArgumentParser(description="Download stocks data.")

parser.add_argument('--region', type=str, required=True)
parser.add_argument('--freq', type=str, required=True)
parser.add_argument('--cor_window', type=int, required=True)
parser.add_argument('--eig_k', type=int, required=True)

# Parse the arguments
args = parser.parse_args()

# HERE PASTE THE API KEY FROM FREDAPI
fred = Fred(api_key='')

if args.region == 'eu':
    print('region: EU')
    spread = fred.get_series('BAMLHE00EHYIOAS') # euro
    tickers = "EBO.DE RAW.DE KBC.BR CBK.DE DBK.DE NDA-SE.ST DANSKE.CO JYSK.CO SYDB.CO BBVA BKT.MC CABK.MC SAB.MC SAN.MC UNI.MC BNP.PA ACA.PA GLE.PA ALPHA.AT EUROB.AT ETE.AT TPEIR.AT OTP.BD A5G.IR BARC.L BIRG.IR BAMI.MI ISP.MI MB.MI BMPS.MI BPE.MI UCG.MI ABN.AS INGA.AS DNB.OL PKO.WA PEO.WA BCP.LS SEB-A.ST SHB-A.ST SWED-A.ST"
    banks_index = pd.read_excel("data/stoxx_banks.xlsx")\
                    .sort_index(ascending=False)\
                    .set_index('Date')
                    
    index = pd.DataFrame(yf.download("^STOXX", start="2000-01-01", group_by='tickers')['Open'])        
                    
elif args.region == 'us':
    print('region: US')
    spread = fred.get_series('BAMLC0A0CM') # usa
    tickers = "BAC BK BCS BMO COF SCHW C CFG DB GS JPM MTB MS NTRS PNC STT TD TFC UBS WFC ALLY AXP DFS FITB HSBC HBAN KEY MUFG PNC RF SAN"
    banks_index = pd.DataFrame(yf.download("^BKX", start="2000-01-01", group_by='tickers')['Open'])        
    index = pd.DataFrame(yf.download("^SPX", start="2000-01-01", group_by='tickers')['Open'])        

spread = pd.DataFrame(spread)
spread.columns = ['spread']   

data_raw = yf.download(tickers, start="2000-01-01", group_by='tickers')

df_rets = data_raw.xs('Close', axis=1, level=1, drop_level=True)\
            .pct_change()\
            .iloc[3:, :]

# subtracting the index returns from the bank returns                
df_rets = df_rets\
            .loc[:banks_index.index[-1],:]\
            .sub(banks_index.pct_change().loc[df_rets.index[0]:,:], axis='columns', fill_value=0)\
            .iloc[:,0:-1]
     
banks_index.columns = ['banks_index']        
index.columns = ['index']

cor_ts = df_rets\
    .fillna(0)\
    .rolling(args.cor_window, min_periods = args.cor_window - 1)\
    .corr()\
    .abs()\
    .groupby(level='Date')\
    .mean()\
    .apply(lambda x: x.mean(), axis=1)

cor_ts = pd.DataFrame(cor_ts)
cor_ts.columns = ['cor']

def eig_connect(x, k):
    e_vals, _ = eig(x)
    return sum(e_vals[0:k])/sum(e_vals)

eigen_ts = df_rets\
    .fillna(0)\
    .rolling(args.cor_window, min_periods = args.cor_window - 1)\
    .cov()\
    .fillna(0)\
    .groupby(level='Date')\
    .apply(lambda x: eig_connect(x, args.eig_k))
    
eigen_ts = pd.DataFrame(eigen_ts)
eigen_ts.columns = ['eig']

granger = pd.DataFrame({'degree': []})

cor_w = 100

for t in range(cor_w, len(df_rets)):
    print(t/len(df_rets))
    window = df_rets.iloc[(t - cor_w):t, :]
    mat = granger_mat(window)
    degree = np.nansum(mat) / (mat.shape[0] * mat.shape[1] - mat.shape[0])
    granger = pd.concat([granger, pd.DataFrame({'degree': [degree]})])


def granger_mat(df):
    
    granger_mat = np.zeros((df.shape[1], df.shape[1]))
    
    for i in range(1, len(df.columns)):
        for j in range(1, len(df.columns)):
            
            if  i == j:
                granger_mat[i,j] = 0
                continue
            
            if all(df.iloc[:,i].isna()) or all(df.iloc[:,j].isna()):
                granger_mat[i,j] = np.nan
                continue
            
            _, pval = granger_cause(df.iloc[:, [i,j]])
            
            granger_mat[i,j] = 1 if pval < 0.05 else 0
            #print(df.columns[i], df.columns[j])
    return granger_mat            
 
def granger_cause(df):
    pair = pd.concat([df.iloc[:, 0], df.iloc[:, 0:2].shift()], axis=1)\
        .dropna()

    model = sm.OLS(pair.iloc[:, 0], sm.add_constant(pair.iloc[:, 1:3]))
    results = model.fit()
    return results.params[1], results.pvalues[1]


df = df_rets\
    .join(cor_ts)\
    .join(eigen_ts)\
    .join(banks_index)\
    .join(spread)\
    .join(index)\
    .reset_index()

if args.freq == 'weekly':       
    print("freq: weekly") 
    first_days = df['Date']\
                .dt.to_period('W')\
                .dt.to_timestamp()\
                .unique()   

    df = df.query('Date in @first_days')
    
    
df['spread_ch'] = df['spread'].pct_change()
df["banks_index"] = df["banks_index"].pct_change()
df['index'] = df['index'].pct_change()

df.to_csv("data/bank_cor.csv") 

# robust yet fragile

# small shock regime
# - no effect of system wide shocks
# higher connectivity higher robustness

# huge shock regime
# - huge effect of system wide shocks
# higher connectivity lower robustness
