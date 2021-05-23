# algorithmic_trading
Master's degree repository

This repository contains codes used for algorithmic trading via Oanda API.\
The repository was created with help of https://www.udemy.com/course/algorithmic-trading-with-python-and-machine-learning/

There is a code which only works having brokerage account in Oanda as connection is necessary.

The goal is to create three separate algorithms for automated trading.\
1st will be based on indicators.\
2nd will be based on price action.\
3rd will be based on machine learning.

Once rules are created then the strategy will be backtested across already downloaded data.\
After that there will be connection established with demo account in Oanda and strategies will be tested in live(demo) market.

This is not a working or complete product.\
All rights reserved.

#################\
Files inside repository\
#################\
Downloading data:\
data_downloader_v2 - can be used to download data from Oanda API and store it offline. It is structured to make multiple query in order to download even 10yr of 3m bars.

Simple Moving Average strategy:\
SMABase and SMAStrategy - modules required for backtesting of SMA strategy\
SMA_cross_v5 - strategy based on cross of two simple moving averages

Price Action strategy:\
PrepareData_v4 - module required to prepare data which is used in PA strategy\
price_action_v6 - iterative backtesting for price action strategy.

Machine Learning - Logistic Regression strategy:\
MLStrategy - module required for Machine Learning strategy\
ML_logistic - logistic regression used to define if next bar will be: increase or decrease. The data is downloaded directly from Oanda API.

Real connection - algorithmic trading:\
RealConn_SMA_v4_BCOUSD - SMA strategy run on oil market\
RealConn_SMA_v4_NATGAS - SMA strategy run on gas market\
RealConn_PA_v4_FR40_EUR - PA strategy run on france 40 market index\
RealConn_PA_v4_SPX500 - PA strategy run on us 500 market index
