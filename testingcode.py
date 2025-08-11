from se762_code import SoccerGame,noisyperturbations,normalizing_probabilities
import numpy as np


A=np.array([[0,0.8,0.7],[0.9,0,0.2],[0.75,0.45,0]])
x0=[0.3,0.3,0.4,0.3,0.3,0.4]
game=SoccerGame(striker_payoff=A)
eq_pts=game.solving_replicator_dynamics(initial_guess=x0)


Noisestarts=noisyperturbations(eq_pts,std=0.2)
t_eval=np.linspace(0,100,2000)
from scipy.integrate import solve_ivp
solutions=solve_ivp(game.replicator_dynamics_time,t_span=(0,100),y0=Noisestarts,t_eval=np.linspace(0,100,2000))

import matplotlib.pyplot as plt
plt.plot(t_eval,solutions.y[0],color='r')
plt.axhline(y=eq_pts[0],ls='--',color='r')

plt.plot(t_eval,solutions.y[1],color='b')
plt.axhline(y=eq_pts[1],ls='--',color='b')

plt.plot(t_eval,solutions.y[2],color='k')
plt.axhline(y=eq_pts[2],ls='--',color='k')

plt.plot(t_eval,solutions.y[3],color='g')
plt.axhline(y=eq_pts[3],ls='--',color='g')

plt.plot(t_eval,solutions.y[4],color='m')
plt.axhline(y=eq_pts[4],ls='--',color='m')

plt.plot(t_eval,solutions.y[5],color='y')
plt.axhline(y=eq_pts[5],ls='--',color='y')

plt.show()