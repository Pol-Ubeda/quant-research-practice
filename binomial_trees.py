import numpy as np
import matplotlib.pyplot as plt

S_0 = 20
strike = 21
r = 0.12
vol = 0.2
N = 150
T = 1 #year

#----------------------
#In our case we are going to assume the steps are for 3 months, so delta_t is 3/12 at each step
#----------------------

class LatticeNode:
    def __init__(self, stock_price):
        self.stock_price = stock_price
        self.option_value = None


def compute_neutral_p(r, dt, u, d):
    e = np.exp(r*dt)
    p = (e-d)/(u-d)
    return p

def compute_option_price(r, dt, p, f_u, f_d):
    expected_option_price = p*f_u + (1-p)*f_d
    e = np.exp(-r*dt)
    current_option_price = e*expected_option_price
    return current_option_price

#----------------------
#Pass n and root node to generate full binomial tree
#----------------------
def build_lattice(n, u, d, S_0, dt):
    S = [[None for i in range(n-t+1)] for t in range(n + 1)]
    p = compute_neutral_p(r,dt,u,d)
        
    for t in range(n+1):
        for i in range (n-t+1):
            stockPrice = S_0 * (u ** i) * (d ** (n-i-t))
            S[n-i-t][i] = LatticeNode(stockPrice)            
            if(t == 0):
                S[n-i-t][i].option_value = max(0, stockPrice - strike)
            else:
                f_u = S[n-i-t][i+1].option_value
                f_d = S[n-i+1-t][i].option_value
                optionPrice = compute_option_price(r,dt,p,f_u,f_d)
                S[n-i-t][i].option_value = optionPrice
    print(f"Steps: {n} ; p = {p} ; price = {S[0][0].option_value}")
    return S

def main():
    evenxpoints = []
    evenypoints = []
    oddxpoints = []
    oddypoints = []
    for n in range (2,N):
        dt = T/n
        
        #We will use CRR parameters, taking volatility into account
        u = np.exp(vol * np.sqrt(dt))
        d = np.exp(-vol * np.sqrt(dt))

        binomialLattice = build_lattice(n,u,d,S_0,dt)
        optionValue = binomialLattice[0][0].option_value
        
        #We plot odd N and even N separately to visualizae oscillation phenomenon
        if(n%2 == 0):
            evenxpoints.append(n)
            evenypoints.append(optionValue)
        else:
            oddxpoints.append(n)
            oddypoints.append(optionValue)
        sumxpoints = [(x+y)/2 for x, y in zip(evenxpoints, oddxpoints)]
        sumypoints = [(x+y)/2 for x, y in zip(evenypoints, oddypoints)]
        
    plt.plot(evenxpoints, evenypoints, color='r', label='odd')
    plt.plot(oddxpoints, oddypoints, color='g', label='even')
    plt.plot(sumxpoints, sumypoints, color='y', label='avg')
    plt.xlabel("N")
    plt.ylabel("Option Price")
    plt.title("Binomial Model Convergence to BSM")
    plt.legend()    
    plt.show()
    return

if __name__ == "__main__":
    main()