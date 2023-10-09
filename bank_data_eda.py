import pandas as pd
import numpy as np

# read xls file

df = pd.read_excel('data/banks_de_2019.xlsx', sheet_name='Results')

# remove column
df = df.drop(columns=['Unnamed: 0'])
df.columns
# change column names to custom
df.columns = ['bank', 'n_employ', 'loans', 'bank_loans', 
              'of_which_other', 'derivs_a', 'cash', 'assets',
              'depo', 'bank_depo', 'derivs_l', 'equity', 'net_interest_rev', 'interest_ret']

# change 'n.a' to NA
df = df.replace('n.a.', np.nan)
df.replace([np.inf, -np.inf], np.nan, inplace=True)

df.apply(lambda x: sum(x.isna()), axis=0)

df = df.drop(columns=['derivs_a', 'derivs_l', 'of_which_other'])
# remove NAs
df = df.dropna()
df.dropna( how="all", inplace=True)
df = df.query("depo != 0")

df['interbank_ratio'] = df.bank_loans / df.assets
# data summary
df['interbank_ratio'].describe().round(3)

np.mean(df['cash'] / df['depo'])

np.mean(df['equity'] / df['assets'])

np.mean(df['bank_loans'] / df['equity'])

df.columns
np.min(df['depo'])
# get only banks with top 10% assets

df = df[df.assets > df.assets.quantile(0.9)]

df['interbank_ratio'].describe().round(3)

sample_df = df.sample(20)

np.round(list(sample_df['depo'] / 100000),2)

np.round(list(sample_df['equity'] / 100000),2)