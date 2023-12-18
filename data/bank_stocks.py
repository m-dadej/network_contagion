import pandas as pd
import yfinance as yf
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

tickers = "EBO.DE RAW.DE KBC.BR CBK.DE DBK.DE NDA-SE.ST DANSKE.CO JYSK.CO SYDB.CO BBVA BKT.MC CABK.MC SAB.MC SAN.MC UNI.MC BNP.PA ACA.PA GLE.PA ALPHA.AT EUROB.AT ETE.AT TPEIR.AT OTP.BD A5G.IR BARC.L BIRG.IR BAMI.MI ISP.MI MB.MI BMPS.MI BPE.MI UCG.MI ABN.AS INGA.AS DNB.OL PKO.WA PEO.WA BCP.LS SEB-A.ST SHB-A.ST SWED-A.ST"

data_raw = yf.download(tickers, start="2000-01-01", group_by='tickers')

df_rets = data_raw.xs('Close', axis=1, level=1, drop_level=True)\
            .pct_change()\
            .iloc[3:, :]
            
df_rets.apply(lambda x: x.isna().sum(), axis=1)

bank_indx = df_rets.apply(lambda x: x.mean(), axis=1)

cor_ts = df_rets\
    .fillna(0)\
    .rolling(200, min_periods = 100)\
    .corr()\
    .groupby(level='Date')\
    .mean()\
    .apply(lambda x: x.mean(), axis=1)

cor_ts = pd.DataFrame(cor_ts)
cor_ts.columns = ['cor']
bank_indx = pd.DataFrame(bank_indx)
bank_indx.columns = ['bank_indx']



df_rets.join(cor_ts).join(bank_indx).to_csv("data/bank_cor.csv") 

# robust yet fragile

# small shock regime
# - no effect of system wide shocks
# higher connectivity higher robustness

# huge shock regime
# - huge effect of system wide shocks
# higher connectivity lower robustness
