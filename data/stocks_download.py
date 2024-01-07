import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from fredapi import Fred
from typing import Dict
import argparse
from numpy.linalg import eig
from sklearn.covariance import LedoitWolf

# Create the parser
parser = argparse.ArgumentParser(description="Download stocks data.")

parser.add_argument('--region', type=str, required=True)
parser.add_argument('--freq', type=str, required=True)
parser.add_argument('--cor_window', type=int, required=True)
parser.add_argument('--eig_k', type=int, required=True)

# Parse the arguments
args = parser.parse_args()

# HERE PASTE THE API KEY FROM FREDAPI
fred = Fred(api_key='18c2830f79155831d5c485d84472811f')

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
