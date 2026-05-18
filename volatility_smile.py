import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt
from scipy.stats import norm

def fetch_real_contract(ticker, target_dte_days=365, contract_type="call"):
    stock = yf.Ticker(ticker)
    S = stock.fast_info["last_price"]
    today = dt.date.today()

    expiries = stock.options
    expiry = min(
        expiries,
        key=lambda e: abs((dt.date.fromisoformat(e) - today).days - target_dte_days)
    )
    expiry_date = dt.date.fromisoformat(expiry)
    T = (expiry_date - today).days / 252

    chain = stock.option_chain(expiry)
    contracts = chain.calls if contract_type == "call" else chain.puts

    print(f"Ticker: {ticker} | S={S:.2f} | Expiry={expiry} | T={T:.3f}yr")
    return S, T, expiry, contracts

def simulate_gbms(S, mu, vol, T, n_steps, n_paths):
    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S
    for t in range(1, n_steps + 1):
        Z = np.random.standard_normal(n_paths)
        paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5*vol**2)*dt + vol*np.sqrt(dt)*Z)
    return paths

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

def compute_vega(S, K, r, current_vol, T):
    d1 = (np.log(S/K) + (r + 0.5*current_vol**2)*T) / (current_vol * np.sqrt(T))
    return S * np.sqrt(T) * norm.pdf(d1)

def newton_raphson(S, K, r, vol0, T, market_price, contract_type):
    current_vol=vol0
    current_option_price = bsm_price(S,K,r,T,vol0, contract_type)
    err = abs(current_option_price - market_price)

    i=0
    while(err > 1e-2):
        vega = compute_vega(S, K, r, current_vol, T)
        new_vol = current_vol - ((current_option_price - market_price) / vega)
        new_option_price = bsm_price(S,K,r,T,new_vol, contract_type)

        current_vol = new_vol
        current_option_price = new_option_price
        err = abs(current_option_price - market_price)
        i += 1
    print(f"NewVol = {current_vol}     |      K = {K}")
    return current_vol

def volatility_smile(contracts, S, r, T, contract_type):
    strikes = []
    ivs = []
    vol0 = 0.3

    for _, row in contracts.iterrows():
        K = row["strike"]

        # skip if bid-ask spread is too wide (illiquid)
        spread = row["ask"] - row["bid"]
        mid = (row["ask"] + row["bid"]) / 2
        if mid > 0 and (spread / mid) > 0.5:  # more than 50% relative spread
            continue
        market_price = mid

        iv = newton_raphson(S, K, r, vol0, T, market_price, contract_type)
        strikes.append(K)
        ivs.append(iv)

    return strikes, ivs
    

def main():
    ticker = 'TSLA'
    r = 0.038
    S, T, expiry, contracts = fetch_real_contract(ticker, target_dte_days=365, contract_type="call")
    call_strikes, call_ivs = volatility_smile(contracts, S, r, T, "call")

    S, T, expiry, contracts = fetch_real_contract(ticker, target_dte_days=365, contract_type="put")
    put_strikes, put_ivs = volatility_smile(contracts, S, r, T, "put")

    plt.figure(figsize=(12, 6))
    plt.plot(call_strikes, call_ivs, label='call IV Smile')
    plt.plot(put_strikes, put_ivs, label='put IV Smile')
    plt.xlabel("Strike Price")
    plt.ylabel("Implied Volatility")
    plt.title("Volatility Smile")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()