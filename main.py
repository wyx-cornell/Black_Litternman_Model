#run the Black-Litternman model and visualize the result
from BL_data_reader import view_data_reader, factor_return_data_reader,\
    factor_loading_data_reader, residual_data_reader
from datetime import datetime, timedelta
import BL_data_reader as BLDR
import numpy as np
import pandas as pd
import timeit
from BL_model import Black_Litterman_Portfolio

#Simple predictor using exponential moving average
def FACTOR_EMA(df, half_life):
    return pd.ewma(df, halflife=half_life)

#using saved predictor calculation result
def dummy_predictor(csv_file):
    _df_factor_return = pd.read_csv(csv_file, sep =',',index_col=0)
    _df_factor_return.index = _df_factor_return.index.map(lambda idx: datetime.strptime(idx,"%Y-%m-%d"))
    return _df_factor_return

def main():
    view_config = {"MA5" : lambda df: FACTOR_EMA(df.shift(1).dropna(axis=0), 5),
                   "RNN": lambda df: dummy_predictor("Prediction/prediction_simple_RNN.csv")}
    print "loading view data reader"
    DR_view = BLDR.view_data_reader("Data/factor_return_w_industry.csv", view_config,
                                error_periods=60, error_method = "rolling_window")
    print "loading factor return data reader"
    DR_factor_return = BLDR.factor_return_data_reader("Data/factor_return_w_industry.csv", look_back_periods=60)
    print "loading residual data reader"
    DR_residual = BLDR.residual_data_reader("Data/residual.csv",look_back_periods=60, threshold = None, min_periods=2, diagonalized=True)#will take a while to load
    DR_residual.set_threshold(0.000001)
    print "loading factor loading data reader"
    DR_factor_loading = BLDR.factor_loading_data_reader("Data/factor_loading_w_industry.csv")#will take a while to load

    data_reader_config = {
        'view': DR_view, #call method return estimation and standard error of each predictors
        'factor_return': DR_factor_return, #call method return prior mean and covariance of factors
        'residual': DR_residual,#call method return covariance of residuals
        'factor_loading': DR_factor_loading #call method return factor loadings of each stock, also return the return of next B day for each stock
        }

    BLP = Black_Litterman_Portfolio(data_reader_config, 1, factor_diagonalized=False)
    print "calculating sample pnl"
    pnl_ts = BLP(datetime(2006,5,1), datetime(2006,5,31))
    pnl_ts.to_csv('pnl_example.csv')
    print 'Example pnl', pnl_ts

if __name__ == "__main__":
    main()