import yfinance as yf
from scipy.stats import norm
import mplfinance as mpf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import sys

def plot_ticker(df):
    mpf.plot(
        df,
        type="candle",
        volume=True,
        mav=(20, 50),
        style="yahoo"
    )

#compute vol of log returns of stock
def estimate_vol(df, col="Close"):
    prices = df[col]
    log_returns = np.log(prices / prices.shift(1)).dropna()
    s = log_returns.std(ddof=1)
    sigma_hat = s / np.sqrt(1/252)
    return sigma_hat

def bsm_price(S0, K, r, T, vol, contract_type="call"):
    d1 = (np.log(S0/K) + (r + vol**2)*T) / (vol * np.sqrt(T))
    d2 = d1 - (vol * np.sqrt(T))    
    n1 = norm.cdf(d1)
    n2 = norm.cdf(d2)
    
    if(contract_type == "call"):
        n1 = norm.cdf(d1)
        n2 = norm.cdf(d2)
        price = S0*n1 - (K * np.exp(-r*T) * n2)
    elif(contract_type == "put"):
        n1 = norm.cdf(-d1)
        n2 = norm.cdf(-d2)
        price = -S0*n1 + (K * np.exp(-r*T) * n2)
    else:
        print("This only admits call or put contract types for now.")
        sys.exit(1)
    return price

def change_in_stock(minS, maxS, vol, K, r, T, n):
    S_vals = np.linspace(minS, maxS)
    C_vals = []
    t = 0
    
    for s in S_vals:
        remaining_time = (n-t)/252
        if(remaining_time == 0): break
        c = bsm_price(s,100,0.12,0.5,vol)
        print(f"The option price for S={s} is {c}")
        
        C_vals.append(c)
        t+=1
    return S_vals, C_vals

def main():
    df = yf.download(
    "AAPL",
    period="6mo",
    interval="1d",
    auto_adjust=True,
    multi_level_index=False
    )
    #This df should contain Index(['Open','High','Low','Close','Adj Close','Volume'], ...)
    sample_size = len(df["Open"])
    vol = estimate_vol(df)
    
    S_vals, C_vals = change_in_stock(60, 140, vol, 100, 0.12, 0.5, sample_size)
    
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    line, = ax.plot([], [], color="#4cc9f0", linewidth=2.5)
    dot, = ax.plot([], [], "o", color="#f72585", markersize=6)

    ax.set_xlim(S_vals.min(), S_vals.max())
    ax.set_ylim(min(C_vals) - 1, max(C_vals) + 1)
    ax.set_title("Black-Scholes Call Price vs Stock Price", fontsize=14)
    ax.set_xlabel("Stock Price S")
    ax.set_ylabel("Option Price C")
    ax.grid(color="white", alpha=0.08)

    def update(frame):
        x = S_vals[:frame]
        y = C_vals[:frame]
        line.set_data(x, y)
        if frame > 0:
            dot.set_data([x[-1]], [y[-1]])
        return line, dot

    ani = FuncAnimation(fig, update, frames=len(S_vals), interval=25, blit=True, repeat=False)
    plt.show()

if __name__ == "__main__":
    main()