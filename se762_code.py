import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


## Setting the game between the striker and the goalie and finding the Nash equilibrium
# Payoff matrix values, mostly arbitrary (if not 1, it means the process is stochastic, meaning the striker can miss even if the GK moves the wrong way or maybe the GK uses a body part to potentially save it)

#Assumptions: 1. probabilities are non-negative and between (0,1)

def normalizing_probabilities(strat):
    strat=np.clip(strat,0,1)
    total=np.sum(strat)
    return strat/total if total!=0 else 1


def noisyperturbations(EQ_pts,std):

# std represents the scale of the initial disturbance being observed
    noise=np.random.normal(loc=0,scale=std,size=len(EQ_pts[0:3]))
    noisydit_ST=EQ_pts[0:3]+noise
    noisydit_ST=normalizing_probabilities(noisydit_ST)
    # print(f'Striker initial condition: {np.round(noisydit,decimals=3)}')

    noise_GK=np.random.normal(loc=0,scale=std,size=len(EQ_pts[3:6]))
    noisydit_GK=EQ_pts[3:6]+noise_GK
    noisydit_GK=normalizing_probabilities(noisydit_GK)
    
    # print(f'GK initial condition: {np.round(noisydit_GK,decimals=3)}')

    return np.concatenate((noisydit_ST,noisydit_GK),axis=None)

# Plotting the KL divergence since it is the Lyapunov candidate

def KL_divergence(EQ_pts,soln_n,t):

    V=[]
    dvdt=[]
    for i in range(len(EQ_pts)):
        v=EQ_pts[i]*np.log(EQ_pts[i]/soln_n[i])
        V.append(v)
        vdot=np.gradient(v,t)
        dvdt.append(vdot)

    VKL=np.sum(V,axis=0)
    dvdt=np.sum(dvdt,axis=0)
    return VKL,dvdt

class SoccerGame:
# zero-sum non cooperative game between striker and goalkeeper
    def __init__(self,striker_payoff,gk_payoff=None):
        self.A=striker_payoff
        if gk_payoff is None:
            self.B = -self.A.T
        else:
            self.B = gk_payoff

    def replicator_dynamics(self,z):
        x=np.array(z[0:3]) #Striker strategy set
        x=np.reshape(x,(3,1))
        y=np.array(z[3:6]) #GK strategy set
        y=np.reshape(y,(3,1))
        dxdt=np.zeros(shape=(3,1)) # Striker replicator dynamics
        dydt=np.zeros(shape=(3,1))
    

        # x and y are column vectors

        Ay=np.matmul(self.A,y) # strriker payoff matrix
        phi=np.matmul(x.T,Ay) # expected value for the striker, average payoff
        Bx=np.matmul(self.B,x)
        phi_2=np.matmul(y.T,Bx)

        for i in range(0,3):
            dxdt[i]=x[i]*(Ay[i]-phi[0,0])
            dydt[i]=y[i]*(Bx[i]-phi_2[0,0])

        return np.concatenate((dxdt,dydt),axis=None)
# print(replicator_dynamics(z=[0.45,0.05,0.5,0.3,0.6,0.1]))


    def solving_replicator_dynamics(self,initial_guess):
        EQ_pts= fsolve(self.replicator_dynamics,x0=initial_guess) #find the equilibrium point of the replicator dynamics
        EQ_pts[:3]=normalizing_probabilities(EQ_pts[:3])
        EQ_pts[3:]=normalizing_probabilities(EQ_pts[3:])
        # print(f"Replicator dynamics equilibrium points striker: {np.array([EQ_pts[0],EQ_pts[1],EQ_pts[2]])}")
        # print(f"Replicator dynamics equilibrium points GK: {np.array([EQ_pts[3],EQ_pts[4],EQ_pts[5]])}")
        self.EQ_pts=EQ_pts
        return EQ_pts




# since the game is non-cooperative, it means that there is at least one Nash equilibrium as proved by Nash


    def replicator_dynamics_time(self,t,z):
    
        return self.replicator_dynamics(z)
    

    ## using Feedback Linearization to make the replicator dynamics asymptotically stable

    def feedback_linearization(self,t,z):
        dxdt_controlled=-0.5*(z-self.EQ_pts)
        return dxdt_controlled

