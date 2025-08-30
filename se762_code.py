import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


## Setting the game between the striker and the goalie and finding the Nash equilibrium
# Payoff matrix values, mostly arbitrary (if not 1, it means the process is stochastic, meaning the striker can miss even if the GK moves the wrong way or maybe the GK uses a body part to potentially save it)

#Assumptions: 1. probabilities are non-negative and between (0,1)

# def uncertainty(delta):
#     return None

def gibbs(u):
    return np.exp(u)/np.sum(np.exp(u))


class SoccerGame:
# zero-sum non cooperative game between striker and goalkeeper
    def __init__(self,striker_payoff,gk_payoff=None):
        self.A=striker_payoff
        if gk_payoff is None:
            self.B = -self.A.T
        else:
            self.B = gk_payoff

    def payoffs(self,x,y):
        striker_payoffs=self.A@y
        gk_payoffs=self.B@x
        return striker_payoffs,gk_payoffs
    
    