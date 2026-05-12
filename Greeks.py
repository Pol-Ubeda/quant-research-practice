import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm

def fetch_ticker(ticker, time_span, interval):
    df = yf.download(
        ticker,
        period=time_span,
        interval = interval,
        auto_adjust = False,
        multi_level_index = False
    )
    return df

def bsm_price(S0, K, r, T, vol, contract_type="call"):
    d1 = (np.log(S0/K) + (r + 0.5*vol**2)*T) / (vol * np.sqrt(T))
    d2 = d1 - (vol * np.sqrt(T))
 
    if contract_type == "call":
        n1 = norm.cdf(d1)
        n2 = norm.cdf(d2)
        price = S0*n1 - (K * np.exp(-r*T) * n2)
    elif contract_type == "put":
        n1 = norm.cdf(-d1)
        n2 = norm.cdf(-d2)
        price = -S0*n1 + (K * np.exp(-r*T) * n2)
    else:
        return None
    return price

def estimate_vol(df, col="Close"):
    prices = df[col]
    log_returns = np.log(prices / prices.shift(1)).dropna()
    s = log_returns.std(ddof=1)
    sigma_hat = s * np.sqrt(252)
    return sigma_hat

def compute_delta(S, K, r, vol, T):
    d1 = (np.log(S/K) + (r + 0.5*vol**2)*T) / (vol * np.sqrt(T))
    return norm.cdf(d1)

def compute_theta(S, K, r, vol, T):
    d1 = (np.log(S/K) + (r + 0.5*vol**2)*T) / (vol * np.sqrt(T))
    d2 = d1 - (vol * np.sqrt(T))
    return (S * norm.pdf(d1) * vol) / (2*np.sqrt(T)) - (r * K * np.exp(-r*T) * norm.cdf(d2))

def compute_gamma(S, K, r, vol, T):
    d1 = (np.log(S/K) + (r + 0.5*vol**2)*T) / (vol * np.sqrt(T))
    return norm.pdf(d1) / (S * vol * np.sqrt(T))

def rebalancing(df, step, K, r, vol, T):
    calls = []
    deltas = []
    prev_delta = 0
    cum_hedge_cost = 0
    hedge_cost = []

    for t in range(0, len(df["Open"]), step):
        if(T-t/252 <= 0): break

        S = df["Open"].iloc[t]
        c = bsm_price(S,K,r,T-t/252,vol,"call")
        delta = compute_delta(S,K,r,vol,T-t/252)
        calls.append(c)
        deltas.append(delta)

        cum_hedge_cost += (delta - prev_delta) * S
        hedge_cost.append(cum_hedge_cost)

        prev_delta = delta  
    return calls, deltas, hedge_cost

def main():
    df = fetch_ticker('AAPL',"1y","1d")
    vol = estimate_vol(df, "Close")
    S = df["Open"].iloc[0]
    K = S + 35
    r = 0.05
    T = 1
    step = 1

    stocks = df["Open"]
    calls = []
    deltas = []
    portfolio_value = []
    
    calls, deltas, hedge_cost = rebalancing(df,step,K,r,vol,T)
    time = [t for t in range(len(df["Close"]))]

    calls_repeated = np.repeat(calls,step)
    deltas_repeated= np.repeat(deltas,step)
    hedge_cost_repeated = np.repeat(hedge_cost,step)

    for i in range(len(time)):
        portfolio_value.append(calls[0]*np.exp(r*i/252) - hedge_cost_repeated[i] + deltas_repeated[i]*stocks[i] - calls_repeated[i])

    print(f"{'Stock':>10} | {'Call price':>12} | {'Delta':>10} | {'Day':>5} | {'Cumulative Cost (00)':>18} | {'Portfolio Value':>14}")
    for i in range(0, len(time), step):
        print(f"{stocks[i]:>10.2f} | {calls_repeated[i]:>12.4f} | {deltas_repeated[i]:>10.4f} | {time[i]:>5} | {hedge_cost_repeated[i]:>18.2f} | {portfolio_value[i]:>14.2f}")

    n = len(time)
    calls_repeated = calls_repeated[:n]
    deltas_repeated = deltas_repeated[:n]
    hedge_cost_repeated = hedge_cost_repeated[:n]

    plt.plot(time, stocks, label='Stock')
    plt.plot(time, portfolio_value, label='Portfolio')
    plt.plot(time, hedge_cost_repeated, label='Hedge')

    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.show()

if __name__ == "__main__":
    main()