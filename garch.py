import numpy as np
import yfinance as yf
import pandas as pd

def fetch_log_returns(ticker: str, period: str = "1y") -> pd.Series:
    raw = yf.download(ticker, period=period, auto_adjust=False,
                      multi_level_index=False)["Close"]
    return np.log(raw / raw.shift(1)).dropna()

def sample_variance(returns):
    var = 0
    n = returns.size
    for i in range(n):
        var += returns.iloc[i] ** 2
    return var / n        

def garch_parameter_estimation(returns):
    n = returns.size
    INTERVAL = np.arange(0.01, 1.0, 0.01)
    current_max_var = -np.inf
    long_run_var = sample_variance(returns) # For this model we assume that the long run avg for var is the variance from historical data

    for a in INTERVAL:
        for b in INTERVAL:
            if(a+b >= 1): break
            current_standardised_residuals = []
            final_standardised_residuals = []
            mle_var = 0
            last_var = returns.iloc[0] ** 2
            # Follow garch model to check parameters for MLE
            for i in range(1,n):
                new_var = (1 - a - b) * long_run_var + a * returns.iloc[i-1]**2 + b * last_var
                mle_var -= np.log(new_var) + returns.iloc[i]**2 / new_var
                current_standardised_residuals.append(returns.iloc[i] / np.sqrt(new_var))
                last_var = new_var
            if(current_max_var < mle_var):
                current_max_var = mle_var
                current_max_parameters = {"omega":1-a-b, "alpha":a, "beta":b, "variance":new_var}
                final_standardised_residuals = current_standardised_residuals.copy()
    return current_max_parameters, final_standardised_residuals

def forecast_variance(max_parameters: dict, returns, t):
    """
    Average forecasted variance over the next t days.
    GARCH(1,1) k-day-ahead forecast: F_k = V + (a+b)^(k-1) * (F_1 - V);
    averaging F_1..F_t over the horizon is just the geometric series sum.
    """
    alpha = max_parameters["alpha"]
    beta = max_parameters["beta"]
    todays_var = max_parameters["variance"]
    persistence = alpha + beta
    long_run_var = sample_variance(returns)

    var = long_run_var + ((1 - persistence**t) / (t * (1 - persistence)) * (todays_var - long_run_var))
    return 252 * var # Return annualized volatility since this uses daily forecasted vols

def mgarch_parameter_estimation(std_residuals: dict, variances: dict):   #sstill have to add weights of portfolio, now we assume they're equal
    df = pd.DataFrame(std_residuals)
    R = df.corr().values
    n = len(std_residuals)
    weights = [np.sqrt(variances[ticker]) / n for ticker in variances.keys()]
    portfolio_var = weights @ R @ weights
    return portfolio_var

def main():
    tickers = ["MSFT","TSLA"]
    residuals_by_ticker = {}
    variance_by_ticker = {}
    t = 30 #days
    for ticker in tickers:
        log_returns = fetch_log_returns(ticker)
        max_parameters, residuals = garch_parameter_estimation(log_returns)
        variance_by_ticker[ticker] = max_parameters["variance"]
        residuals_by_ticker[ticker] = residuals
        print(f"GARCH model for ticker {ticker}")
        print(f'omega = {max_parameters["omega"]}    alpha = {max_parameters["alpha"]}    beta = {max_parameters["beta"]}')
        print(f"Today's volatility = {np.sqrt(max_parameters['variance'])}")

        forecasted_volatility = np.sqrt(forecast_variance(max_parameters, log_returns, t))
        print(f"The annual volatility averaged from forecasted variances using GARCH is: {forecasted_volatility}")
    mgarch_parameter_estimation(residuals_by_ticker,variance_by_ticker)

if __name__ == "__main__":
    main()