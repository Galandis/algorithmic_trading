import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
plt.style.use("seaborn")

class LogisticStrategy():
    ''' Class for the vectorized backtesting of Machine Learning-based trading strategies (Classification).

    Attributes
    ==========
    symbol: str
        ticker symbol with which to work with
    start: str
        start date for data retrieval
    end: str
        end date for data retrieval
    spread: float
        transaction costs per trade

    Methods
    =======
    get_data:
        retrieves and prepares the raw data
    select_data:
        selects a sub-set of the data
    prepare_features:
        prepares the features for the model fitting/prediction 
    fit_model:
        fitting the ML model
    test_strategy:
        runs the backtest for the ML-based strategy
    plot_results:
        plots the performance of the strategy compared to buy and hold
    '''

    def __init__(self, symbol, timeframe, cost):
        self.symbol = symbol
        self.timeframe = timeframe
        self.cost = cost
        self.model = LogisticRegression(C = 1e6, max_iter = 100000, multi_class = "ovr")
        self.results = None
        self.get_data()
    
    def __repr__(self):
        rep = "MLBacktester(symbol = {}, timeframe = {}, trading_cost = {})"
        return rep.format(self.symbol, self.timeframe, self.cost)
                             
    def get_data(self):
        ''' Download static data from folder.
        '''
        raw = pd.read_csv("Data/{}_{}.csv".format(self.symbol, self.timeframe), parse_dates = ["time"], index_col = "time")
        raw = raw.drop(['o','h','l'], axis = 1)
        raw = raw.rename(columns={"c":"price"})
        raw["returns"] = np.log(raw.price / raw.price.shift(1))
        # Distance between rolling min/max prices and current price
        raw["min"] = raw.price.rolling(20).min() / raw.price - 1
        raw["max"] = raw.price.rolling(20).max() / raw.price - 1
        # Rolling mean returns
        raw["mom"] = raw["returns"].rolling(3).mean()
        raw = raw.dropna()
        
        self.data = raw.copy(deep=True)
        self.train = self.data[:"2021-01-01"] # specifing training set
        self.test = self.data["2021-01-01":] # specifing test set
        return raw
    
    def prepare_features(self, dataset):
        ''' Prepares the feature columns for the fitting/prediction steps.
        '''
        self.data_subset = dataset
        features = ["price", "volume", "min", "max", "mom"]
        self.feature_columns = []
        for f in features:
            for lag in range(1, self.lags + 1):
                col = "{}_lag_{}".format(f, lag)
                self.data_subset[col] = self.data_subset[f].shift(lag)
                self.feature_columns.append(col)
        self.data_subset.dropna(inplace=True)
        
    def fit_model(self, dataset):
        ''' Fitting the ML Model.
        '''
        self.prepare_features(dataset)
        self.model.fit(self.data_subset[self.feature_columns], np.sign(self.data_subset.loc[:,("returns")]))
        
    def test_strategy(self, lags=5):
        ''' Backtests the trading strategy.
        '''
        self.lags = lags
        
        # fit the model on the training set
        self.fit_model(self.train)
        
        # prepare the test set
        self.prepare_features(self.test)
        
        # make predictions on the test set
        prediction = self.model.predict(self.data_subset[self.feature_columns])
        self.data_subset.loc[:,("pred")] = prediction
        
        # calculate Strategy Returns
        self.data_subset.loc[:,("strategy")] = self.data_subset.loc[:,("pred")] * self.data_subset.loc[:,("returns")]
        
        # determine when a trade takes place
        self.data_subset.loc[:,("trades")] = self.data_subset.loc[:,("pred")].diff().fillna(0).abs()
        
        # subtract transaction costs from return when trade takes place
        self.data_subset.strategy = self.data_subset.strategy - self.data_subset.trades * self.cost
        
        # calculate cumulative returns for strategy & buy and hold
        self.data_subset.loc[:,("creturns")] = self.data_subset.loc[:,("returns")].cumsum().apply(np.exp)
        self.data_subset.loc[:,("cstrategy")] = self.data_subset.loc[:,('strategy')].cumsum().apply(np.exp)
        self.results = self.data_subset
        
        # absolute performance of the strategy
        perf = self.results["cstrategy"].iloc[-1]
        # out-/underperformance of strategy
        outperf = perf - self.results["creturns"].iloc[-1]
        
        return round(perf, 6), round(outperf, 6)
        
    def plot_results(self):
        ''' Plots the cumulative performance of the trading strategy
        compared to buy and hold.
        '''
        if self.results is None:
            print("No results to plot yet. Run a strategy.")
        else: 
            title = "{} | Trading Cost = {}".format(self.symbol, self.cost)
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))
