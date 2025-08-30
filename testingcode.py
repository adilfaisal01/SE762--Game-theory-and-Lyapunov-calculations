from se762_code import SoccerGame,gibbs
import numpy as np
import matplotlib.pyplot as plt
import nashpy as nash

A=np.array([[0,0.8,0.7],[0.9,0,0.2],[0.75,0.45,0]])
soccergame=SoccerGame(striker_payoff=A)
B=soccergame.B
Game=nash.Game(A,B)

nash_eq=Game.support_enumeration() #Nash equilibrium of the game
optimalpoints=list(nash_eq)[0][:][0]

x=np.array([[0.3],[0.3],[0.4]])
y=np.array([[0.3],[0.3],[0.4]])

sp,gp=soccergame.payoffs(x,y) #striker and goalkeeper payoffs

T=60 #simulation time
n=100#number of simulation steps
time_array=np.linspace(0,T,n)
u=np.zeros((3,1))
v=np.zeros((3,1))
xhistory=[]
yhistory=[]
dt=T/n
xhistory.append(x.flatten())
yhistory.append(y.flatten())


for t in time_array:
    strikerpayoff,goalkeeperpayoff=soccergame.payoffs(x,y) #payoff, 3x1 which will be input as P=udot in Mabrok's paper
    # T=P-1*x
    # integrator (1/s)
    u=u+strikerpayoff*dt
    v=v+goalkeeperpayoff*dt

    #strategy being updated
    x=gibbs(u)
    y=gibbs(v)

    # storing the history
    xhistory.append(x.flatten())
    yhistory.append(y.flatten())

xhistory=np.array(xhistory)
yhistory=np.array(yhistory)
time_array= np.concatenate([[0], time_array])

plt.figure()
plt.plot(time_array,yhistory[:,0],label='rock')
plt.plot(time_array,yhistory[:,1],label='paper')
plt.plot(time_array,yhistory[:,2],label='scissors')
plt.xlim((-0.1,61))
plt.show()

plt.figure()
plt.plot(time_array,xhistory[:,0],label='rock')
plt.plot(time_array,xhistory[:,1],label='paper')
plt.plot(time_array,xhistory[:,2],label='scissors')
plt.xlim((-0.1,61))
plt.show()