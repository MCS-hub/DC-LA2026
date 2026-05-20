import numpy as np
import matplotlib.pyplot as plt

def update_step(Xk, lam, gamma, d, grad_f, prox_r1, prox_r2, inv_temp=1.0):
    Z = np.random.randn(d)  # Gaussian noise
    term1 = Xk
    term2 = -(gamma * lam) / (gamma + lam) * grad_f(Xk)
    term3 = -(gamma / (gamma + lam)) * prox_r2(Xk, lam)
    term4 = (lam * np.sqrt(2 * gamma / inv_temp)) / (gamma + lam) * Z
    
    inner = ((lam + gamma) / lam) * Xk \
            - gamma * grad_f(Xk) \
            - (gamma / lam) * prox_r2(Xk, lam) \
            + np.sqrt(2 * gamma / inv_temp) * Z
    
    term5 = (gamma / (gamma + lam)) * prox_r1(inner, lam + gamma)
    
    return term1 + term2 + term3 + term4 + term5

def DC_LA(X0, n_samples, burn_in, lam, gamma, d, grad_f, prox_r1, prox_r2, inv_temp=1.0):
    samples = []
    X = X0.copy()
    
    for k in range(n_samples + burn_in):
        X = update_step(X, lam, gamma, d, grad_f, prox_r1, prox_r2, inv_temp=inv_temp)
        if k >= burn_in:
            samples.append(X)
    return np.array(samples)

def update_step_v2(Xk, lam, gamma, grad_f, prox_r1, prox_r2, inv_temp=1.0):
    m, n = Xk.shape
    Z = np.random.randn(m, n)  # Gaussian noise
    term1 = Xk
    term2 = -(gamma * lam) / (gamma + lam) * grad_f(Xk)
    term3 = -(gamma / (gamma + lam)) * prox_r2(Xk, lam)
    term4 = (lam * np.sqrt(2 * gamma / inv_temp)) / (gamma + lam) * Z
    
    inner = ((lam + gamma) / lam) * Xk \
            - gamma * grad_f(Xk) \
            - (gamma / lam) * prox_r2(Xk, lam) \
            + np.sqrt(2 * gamma / inv_temp) * Z
    
    term5 = (gamma / (gamma + lam)) * prox_r1(inner, lam + gamma)
    
    return term1 + term2 + term3 + term4 + term5

def DC_LA_v2(X0, n_samples, burn_in, lam, gamma, grad_f, prox_r1, prox_r2, inv_temp=1.0):
    samples = []
    X = X0.copy()
    
    for k in range(n_samples + burn_in):
        X = update_step_v2(X, lam, gamma, grad_f, prox_r1, prox_r2, inv_temp=inv_temp)
        if k >= burn_in:
            samples.append(X)
    return np.array(samples)

def PSGLA(X0, n_samples, burn_in, gamma, d, grad_f, prox_r, inv_temp=1.0):
    samples = []
    X = X0.copy()
    
    for k in range(n_samples + burn_in):
        Z = np.random.randn(d)  # Gaussian noise
        X = prox_r(X - gamma * grad_f(X) + np.sqrt(2 * gamma / inv_temp) * Z, gamma)
        if k >= burn_in:
            samples.append(X)
    return np.array(samples)

def ISTA(X0, maxit, gamma, grad_f, prox_r):
    X = X0.copy()
    for k in range(maxit):
        X = prox_r(X - gamma * grad_f(X), gamma)
    return X

def ULA_s(X0, n_samples, burn_in, lam, gamma, d, grad_f, prox_r1, prox_r2, inv_temp=1.0):
    samples = []
    X = X0.copy()
    
    for k in range(n_samples + burn_in):
        Z = np.random.randn(d)  # Gaussian noise
        X = X - gamma * grad_f(X) + (gamma / lam) * prox_r1(X, lam) - (gamma / lam) * prox_r2(X, lam) + np.sqrt(2 * gamma / inv_temp) * Z
        if k >= burn_in:
            samples.append(X)
    return np.array(samples)


def ULA_ns(X0, n_samples, burn_in, gamma, d, grad_f, grad_r1, grad_r2, inv_temp=1.0):
    samples = []
    X = X0.copy()
    
    for k in range(n_samples + burn_in):
        Z = np.random.randn(d)  # Gaussian noise
        X = X - gamma * grad_f(X) - gamma * grad_r1(X) + gamma * grad_r2(X) + np.sqrt(2 * gamma / inv_temp) * Z
        if k >= burn_in:
            samples.append(X)
    return np.array(samples)