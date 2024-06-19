from matplotlib import figure
import numpy as np
from numpy.polynomial import polynomial as P
from scipy.optimize import curve_fit, minimize
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd


# Read in csv data
fname = "/Users/dzine/Downloads/E19/Final Project/E19 - MasterData.csv"
df = pd.read_csv(fname, header=0, usecols=[
                 "PA_POP", "Days after 10-01-2020", "Confirmed", "Deaths", "Recovered", "Active"])


# 12,801,989 total population of PA on 10/1/2021 -- CONSTANT
N = df['PA_POP'][0]

# Infected as proportion
I = np.array(df['Confirmed'][0:157]) / N
# print(I)
# Inew = np.zeros(731)
# temp = np.array(df['Confirmed'] / N)
# Inew[0:315] = temp
# print(Inew)

# Can combine deaths and recovered due to assumptions, keep as proportion
R = np.array(df['Deaths'][0:157] + df['Recovered'][0:157]) / N
# print(R)

# S + I + R = N, and since Active = N - R, S = Active - I, keep as proportion
S = np.array(N - df['Confirmed'][0:157] - R) / N
# print(S)

# Create array of time data
t_start = 1
t_end = df["Days after 10-01-2020"][len(S)-1]

t = np.linspace(t_start, t_end, len(S))
# print(t)

# # ------------------------------------------------------------------------#

# SIR model differential equations


def deriv(x, t, beta, gamma):
    s, i, r = x
    dsdt = -beta * s * i
    didt = beta * s * i - gamma * i
    drdt = gamma * i
    return [dsdt, didt, drdt]

# Function to calculate S, I, R using the SIR model equations


def sir_solution(t, beta, gamma, S0, I0, R0):
    x_initial = S0, I0, R0
    soln = odeint(deriv, x_initial, t, args=(beta, gamma))
    s, i, r = soln.T
    return s, i, r

# Given initial guess of beta/gamma params, find mean squared error of
# sir model estimation to actual sir data


def mse(params):
    beta, gamma = params
    S0, I0, R0 = S[0], I[0], R[0]
    S_model, I_model, R_model = sir_solution(t, beta, gamma, S0, I0, R0)
    return np.mean((S_model - S)**2 + (I_model - I)**2 + (R_model - R)**2)


# Initial guess for parameters
beta_guess = 0.2
gamma_guess = 0.1

param_guess = np.array([beta_guess, gamma_guess])

result = minimize(mse, param_guess, method='Powell')

beta_optimized = result.x[0]
# print(beta_optimized)
gamma_optimized = result.x[1]
# print(gamma_optimized)

x_initial = S[0], I[0], R[0]
soln = odeint(deriv, x_initial, t, args=(beta_optimized, gamma_optimized))
s, i, r = soln.T
# print(soln)

fig = plt.figure(1)
plt.plot(t, S)
plt.plot(t, s, 'r--')
plt.legend(['S(t)', 'S_approx(t)'], loc="upper right")
plt.grid()
plt.show()

fig2 = plt.figure(2)
plt.plot(t, I)
plt.plot(t, i, 'g--')
plt.plot(t, R)
plt.plot(t, r, 'b--')
plt.grid()
plt.legend(['I(t)', 'I_approx(t)', 'R(t)', 'R_approx(t)'], loc="upper right")
plt.show()

# -------------------------------------------------------------------------------------#
# RUN AFTER OBTAINING OPTIMIZED PARAMS
# -------------------------------------------------------------------------------------#

beta_optimized = 0.019777253191273228
gamma_optimized = 0.00826502898050536

t_ext = np.linspace(t_start, 731, 731)
# t_ext = np.linspace(t_start, len(df), len(df))
# print(t_ext)
# print(len(df))

x_initial = S[0], I[0], R[0]
soln = odeint(deriv, x_initial, t_ext, args=(beta_optimized, gamma_optimized))
s, i, r = soln.T


fig = plt.figure(3)
# plt.plot(t, S)
plt.plot(t_ext, s, 'r--')
# plt.plot(t_ext, Inew)
# plt.plot(t_ext, I)
plt.plot(t_ext, i, 'g--')
# plt.plot(t, R)
plt.plot(t_ext, r, 'b--')
plt.legend(['S_approx(t)', 'I_approx(t)', 'R_approx(t)'], loc="upper right")
plt.grid()
plt.show()
