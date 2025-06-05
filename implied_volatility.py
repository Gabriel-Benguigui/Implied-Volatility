import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

# Modèle Black-Scholes
def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Fonction d’inversion pour trouver la volatilité implicite
def implied_volatility(market_price, S, K, T, r, option_type='call'):
    objective = lambda sigma: black_scholes_price(S, K, T, r, sigma, option_type) - market_price
    try:
        return brentq(objective, 1e-6, 5.0, maxiter=1000)
    except ValueError:
        return np.nan

# Exemple
S = 100
K = 100
T = 0.5
r = 0.01
market_price = 5.5

iv_call = implied_volatility(market_price, S, K, T, r, option_type='call')
iv_put = implied_volatility(market_price, S, K, T, r, option_type='put')

print(f"Volatilité implicite (call) : {iv_call:.4%}")
print(f"Volatilité implicite (put)  : {iv_put:.4%}")
