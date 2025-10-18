from se762_code import SoccerGame,gibbs
import numpy as np
import matplotlib.pyplot as plt
import nashpy as nash

A=np.array([[0,0.8,0.7],[0.9,0,0.2],[0.75,0.45,0]])+np.random.normal(loc=0,scale=0.05,size=(3,3))
# A=np.array([[0,1,1],[1,0,1],[1,1,0]]) #perfect symmetric matrix, toy test
soccergame=SoccerGame(striker_payoff=A)
B=soccergame.B
Game=nash.Game(A,B)

nash_eq=Game.support_enumeration() #Nash equilibrium of the game
optimalpoints=np.reshape(list(nash_eq)[0],(6,1))
#initial strategies (make it more randomly generated)
x=np.reshape(np.array(optimalpoints[0:3]),shape=(3,1))+np.random.normal(loc=0.05,scale=0.10,size=(3,1))
y=np.reshape(np.array(optimalpoints[3:6]),shape=(3,1))+np.random.normal(loc=0.05,scale=0.10,size=(3,1))

T=100 #simulation time
n=100000#number of simulation steps
time_array=np.linspace(0,T,n)
u = np.log(x).reshape(3,1)
v = np.log(y).reshape(3,1)
xhistory=[]
yhistory=[]
dt=T/n
xhistory.append(x.flatten())
yhistory.append(y.flatten())

for t in time_array:
    strikerpayoff,goalkeeperpayoff=soccergame.payoffs(x,y) #payoff, 3x1 which will be input as P=udot in Mabrok's paper
    T1,T2=soccergame.feedbackgain([1,1],x,y)
    u = u+T1*dt
    v = v+T2*dt
    #strategy being updated
    x=gibbs(u) #striker strategy
    y=gibbs(v) #gk strategy

    # storing the history
    xhistory.append(x.flatten())
    yhistory.append(y.flatten())

xhistory=np.array(xhistory)
yhistory=np.array(yhistory)
time_array= np.concatenate([[0], time_array])

plt.figure()
plt.plot(time_array,yhistory[:,0],label='left')
plt.plot(time_array,yhistory[:,1],label='center')
plt.plot(time_array,yhistory[:,2],label='right')
plt.xlim((-0.1,T+1))
plt.title('gk, control')
plt.legend()
plt.show()

plt.figure()
plt.plot(time_array,xhistory[:,0],label='left')
plt.plot(time_array,xhistory[:,1],label='center')
plt.plot(time_array,xhistory[:,2],label='right')
plt.xlim((-0.1,T+1))
plt.title('striker control')
plt.legend()
plt.show()