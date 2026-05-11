import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

def fetch_ticker(ticker, time_span, interval):
    df = yf.download(
        ticker,
        period=time_span,
        interval = interval,
        auto_adjust = False,
        multi_level_index = False
    )
    return df

def estimate_vol(df):
    prices = df["Close"]
    log_returns = np.log(prices / prices.shift(1)).dropna()
    s = log_returns.std(ddof=1)
    vol = s * np.sqrt(252)
    return vol

def random_walk(S0, vol, r, T):
    current_price = S0
    walk = []
    t = 1/252
    
    for day in range(T):
        z = np.random.normal()
        next_price = current_price * np.exp((r-(vol**2)/2)*t + vol*np.sqrt(t)*z)
        walk.append(next_price)
        current_price = next_price
    return walk

def main():
    df = fetch_ticker('AAPL',"1y","1d")
    vol = estimate_vol(df)
    S0 = df["Open"].iloc[0]
    r = 0.05
    T = 30
    K = S0+5
    final_option_values = []
    ITERATIONS = 100
    final_call_values = []
    final_put_values = []

    for n in range(ITERATIONS):
        Ypoints = random_walk(S0,vol,r,T)
        Xpoints = [t for t in range(len(Ypoints))]
        plt.plot(Xpoints,Ypoints,alpha=0.6)

        final_call_values.append(np.maximum(Ypoints[-1] - K, 0)) #list of option values for a call option
        final_put_values.append(np.maximum(K - Ypoints[-1], 0)) #list of option values for a put option
    
    #compute risk neutral option price by discounting average end prices from sim
    call_price = np.exp(-r*T/252) * np.mean(final_call_values)
    put_price = np.exp(-r*T/252) * np.mean(final_put_values)

    print(f"The initial stock price is: S0={S0}")
    print(f"Volatility estimated is: {vol}")
    print(f"The call option price is: c = {call_price}")
    print(f"The put option price is: p = {put_price}")

    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.title("Monte Carlo Simulations")
    plt.show()

if __name__ == "__main__":
    main()