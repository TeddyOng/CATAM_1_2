import pandas as pd
from sympy.solvers import solve
from sympy import Symbol
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import linregress
import numpy as np
import random

def f(x,y):
    return -4*y+3*np.exp(-x)

def g(x):
    return np.exp(-x)-np.exp(-4*x)

def euler(f,h,x0,Y0,n):
    Y = Y0
    x = x0
    listY = [Y0]
    listx = [x0]

    for i in range(0,n):
        Y = Y + h * f(x,Y)
        x = x + h
        listY.append(Y)
        listx.append(x)

    return (listx,listY)

def leapfrog(f,h,x0,Y0,n):
    Y1 = Y0 + h * f(x0,Y0)
    x1 = x0 + h
    listY = [Y0,Y1]
    listx = [x0,x1]
    Y = Y1
    x = x1

    for i in range(0,n-1):
        Y = listY[-2] + 2 * h * f(x,Y)
        x = x + h
        listY.append(Y)
        listx.append(x)

    return (listx,listY)

def RK4(f,h,x0,Y0,n):

    Y = Y0
    x = x0
    listY = [Y0]
    listx = [x0]

    for i in range (0,n):
        k1 = h * f(x,Y)
        k2 = h * f(x + h/2, Y + k1 / 2)
        k3 = h * f(x + h/2, Y + k2 / 2)
        k4 = h * f(x + h, Y + k3)

        Y = Y + (k1 + 2 * k2 + 2 * k3 + k4)/6

        x = x + h

        listY.append(Y)
        listx.append(x)

    return (listx, listY)

def Error(g,method,f,h,x0,Y0,n):
    result = method(f,h,x0,Y0,n)
    xplotrange = [x0 + i*h for i in range(0,n+1)]

    return (xplotrange,[result[1][i]-g(xplotrange[i]) for i in range(0,n+1)])

# Section 2.1 Question 1

x0 = 0
Y0 = g(x0)
xEnd = 10

Gamma = []

for i in range(0,4):
    h = [0.4, 0.2, 0.1, 0.05][i]
    n = int((xEnd-x0)/h)

    PlotFileNames = ['Q21aA.csv','Q21bA.csv','Q21cA.csv','Q21dA.csv']
    ErrorFileNames = ['Q21aE.csv','Q21bE.csv','Q21cE.csv','Q21dE.csv']

    result1 = leapfrog(f,h,x0,Y0,n)

    plotrange = np.arange(x0,xEnd+h/2,h)

    np.savetxt(PlotFileNames[i], np.transpose(result1), delimiter = ',')

    errorresult = Error(g,leapfrog,f,h,x0,Y0,n)

    np.savetxt(ErrorFileNames[i], np.transpose(errorresult), delimiter = ',')

    df = pd.DataFrame({
                'x': [x0 + i*h for i in range(5,n)],
                'logE': [np.log(np.abs(errorresult[1][i])) for i in range(5,n)]
                })

    stats1 = linregress(df.x,df.logE)
    m1 = stats1.slope

    Gamma.append(m1)

    """
    plt.figure(figsize=(5,5))
    plt.plot(df.x,df.logE,marker='.')
    plt.show()
    """

print("the diverging coefficient are:")

print(Gamma)


# Section 2 Question 3

h = 0.4
x0 = 0
Y0 = g(x0)
xEnd = 4
n = int((xEnd-x0)/h)

np.savetxt("Q2.3Euler.csv", np.transpose(euler(f,h,x0,Y0,n)), delimiter = ',')
np.savetxt("Q2.3RK4.csv", np.transpose(RK4(f,h,x0,Y0,n)), delimiter = ',')



# Section 2 Question 4
EulerErrorList = []
LFErrorList = []
RK4ErrorList = []


EulerLogError = []
LFLogError = []
RK4LogError = []
for i in range(0,16):
    h = 0.4 / 2**i
    n = 2**i

    result1 = euler(f,h,x0,Y0,n)
    result2 = leapfrog(f,h,x0,Y0,n)
    result3 = RK4(f,h,x0,Y0,n)

    EulerError = result1[1][-1] - g(0.4)
    LFError = result2[1][-1] - g(0.4)
    RK4Error = result3[1][-1] - g(0.4)

    EulerErrorList.append((h,EulerError))
    LFErrorList.append((h,LFError))
    RK4ErrorList.append((h,RK4Error))

    EulerLogError.append((np.log(h),np.log(np.abs(EulerError))))
    LFLogError.append((np.log(h),np.log(np.abs(LFError))))
    RK4LogError.append((np.log(h),np.log(np.abs(RK4Error))))

df1 = pd.DataFrame({
        'logh': [EulerLogError[i][0] for i in range(2,10)],
        'logE': [EulerLogError[i][1] for i in range(2,10)]
        })

stats1 = linregress(df1.logh,df1.logE)
m1 = stats1.slope

df2 = pd.DataFrame({
        'logh': [LFLogError[i][0] for i in range(2,10)],
        'logE': [LFLogError[i][1] for i in range(2,10)]
        })

stats2 = linregress(df2.logh,df2.logE)
m2 = stats2.slope

df3 = pd.DataFrame({
        'logh': [RK4LogError[i][0] for i in range(2,10)],
        'logE': [RK4LogError[i][1] for i in range(2,10)]
        })

stats3 = linregress(df3.logh,df3.logE)
m3 = stats3.slope

ErrorO = [m1,m2,m3]
print("the slopes (order of accuracy) are:")
print(ErrorO)


np.savetxt("Q2.4LogEuler.csv", EulerLogError, delimiter = ',')
np.savetxt("Q2.4LogLF.csv", LFLogError, delimiter = ',')
np.savetxt("Q2.4LogRK4.csv", RK4LogError, delimiter = ',')

np.savetxt("Q2.4Euler.csv", EulerErrorList, delimiter = ',')
np.savetxt("Q2.4LF.csv", LFErrorList, delimiter = ',')
np.savetxt("Q2.4RK4.csv", RK4ErrorList, delimiter = ',')

"""
plt.figure(figsize=(5,5))
#   plt.plot(xplot,yplot,marker='x')
plt.plot(xplot1,yplot1,marker='+')
#   plt.plot(xplot2,yplot2,marker='o')
plt.plot(plotrange,[g(x) for x in plotrange],color="red")
plt.xlabel("x", fontsize=16)
plt.ylabel("Y", fontsize=16)
plt.show()
"""


#   LogError = ([x0 + i*h for i in range(1,n)],[np.log(np.abs(errorresult[1][i])) for i in range(1,n)])

"""
plt.figure(figsize=(5,5))
plt.plot(errorresult[0],errorresult[1],marker='.')
plt.show()
"""


# Section 2.1 Question 2


# Section 3 Question 6


delta = 0
Omega = 1
a = 1
gamma = 1
omega = np.sqrt(3)

k = np.sqrt(4 * Omega **2 - gamma**2)

A = (Omega**2 - omega**2) * a / ((Omega**2 - omega**2)**2 + gamma ** 2 * omega ** 2)

B = - gamma * omega * a / ((Omega**2 - omega**2)**2 + gamma**2 * omega**2) 

D = - B

C = ((gamma * D / 2) - omega * A) / (k / 2)

def f(t,y):
    return np.array([y[1],- gamma * y[1] - delta**3 *  (y[0])**2 * y[1] - Omega**2 * y[0] + a * np.sin(omega * t)])


# Section 3 Question 5 Analytical Solution
def g(t):
    return A * np.sin(omega * t) + B * np.cos(omega * t) + np.exp(- gamma * t / 2) * (C * np.sin(k * t / 2) + D * np.cos(k * t / 2))

"""
def gg(t):
    return omega* A * np.cos(omega * t) - omega * B * np.sin(omega * t) - (gamma / 2) * np.exp(- gamma * t / 2) * (C * np.sin(k * t / 2) + D * np.cos(k * t / 2)) + np.exp(- gamma * t / 2) * (k/2) * (C * np.cos(k * t / 2) - D * np.sin(k * t / 2))

def ggg(t):
    return - omega**2 * A * np.sin(omega * t) - omega**2 * B * np.cos(omega * t) + (gamma / 2)**2 * np.exp(- gamma * t / 2) * (C * np.sin(k * t / 2) + D * np.cos(k * t / 2)) - 2 * (gamma/2) * np.exp(- gamma * t / 2) * (k/2) * (C * np.cos(k * t / 2) - D * np.sin(k * t / 2)) - np.exp(- gamma * t / 2) * (k/2)**2 * (C * np.sin(k * t / 2) + D * np.cos(k * t / 2))

def h(t):
    return (-2/7) * np.sin(omega * t) + (-np.sqrt(3)/7) * np.cos(omega * t) + np.exp(- gamma * t / 2) * ((5/7) * np.sin(k * t / 2) + (np.sqrt(3)/7) * np.cos(k * t / 2))
"""

def RK4plotError(g,f,h,x0,Y0,n):
    return [RK4(f,h,x0,Y0,n)[1][i][0] - g(i * h) for i in range(0,n+1)]


result1 = [RK4(f,0.4,0,[0,0],25)[1][i][0] for i in range(0,26)]
result2 = [RK4(f,0.2,0,[0,0],50)[1][i][0] for i in range(0,51)]
result3 = [RK4(f,0.1,0,[0,0],100)[1][i][0] for i in range(0,101)]

#   np.savetxt("Q3.6g.csv", np.transpose((np.arange(0,10.005,0.01),[g(x) for x in np.arange(0,10.005,0.01)])), delimiter = ',')
np.savetxt("Q3.6a.csv", np.transpose((np.arange(0,10.05,0.4),result1)), delimiter = ',')
np.savetxt("Q3.6b.csv", np.transpose((np.arange(0,10.05,0.2),result2)), delimiter = ',')
np.savetxt("Q3.6c.csv", np.transpose((np.arange(0,10.05,0.1),result3)), delimiter = ',')

"""
plt.figure(figsize=(5,5))
plt.plot(np.arange(0,10.05,0.4),result1,marker='.')
plt.plot(np.arange(0,10.05,0.2),result2,marker='+')
plt.plot(np.arange(0,10.05,0.1),result3,marker='x')
plt.plot(np.arange(0,10.05,0.1),[g(x) for x in np.arange(0,10.05,0.1)],color = "red")
#   plt.plot(np.arange(0,10.05,0.1),[h(x) for x in np.arange(0,10.05,0.1)],color = "orange")
#   plt.plot(np.arange(0,10.05,0.1),[ggg(x) + gamma * gg(x) + Omega**2 * g(x) for x in np.arange(0,10.05,0.1)],color = "blue")
#   plt.plot(np.arange(0,10.05,0.1),[a * np.sin(omega * x) for x in np.arange(0,10.05,0.1)],color = "black")
plt.show()
"""

# Plotting the Errors
"""
plt.figure(figsize=(5,5))
plt.plot(np.arange(0,10.05,0.4),RK4plotError(g,f,0.4,0,[0,0],25),marker='.')
plt.plot(np.arange(0,10.05,0.2),RK4plotError(g,f,0.2,0,[0,0],50),marker='+')
plt.plot(np.arange(0,10.05,0.1),RK4plotError(g,f,0.1,0,[0,0],100),marker='x')
plt.show()
"""

np.savetxt("Q3.6Errora.csv", np.transpose((np.arange(0,10.05,0.4),RK4plotError(g,f,0.4,0,[0,0],25))), delimiter = ',')
np.savetxt("Q3.6Errorb.csv", np.transpose((np.arange(0,10.05,0.2),RK4plotError(g,f,0.2,0,[0,0],50))), delimiter = ',')
np.savetxt("Q3.6Errorc.csv", np.transpose((np.arange(0,10.05,0.1),RK4plotError(g,f,0.1,0,[0,0],100))), delimiter = ',')

"""

# Section 3 Question 7
omega = 1
h = 0.2
n = int(40/h)
GammaFileNames = [["Q3.71a.csv","Q3.71b.csv","Q3.71c.csv","Q3.71d.csv"],["Q3.72a.csv","Q3.72b.csv","Q3.72c.csv","Q3.72d.csv"]]
for j in range(0,2):
    omega = j+1
    for i in range(0,4):
        gamma = [0.25,0.5,1.0,1.9][i]
        result1 = [RK4(f,h,0,[0,0],n)[1][i][0] for i in range(0,n+1)]
        np.savetxt(GammaFileNames[j][i], np.transpose(([i*h for i in range(0,n+1)],result1)), delimiter = ',')

#    plt.figure(figsize=(5,5))
#    plt.plot([i*h for i in range(0,n+1)],result1,marker='.')
#    plt.plot(np.arange(0,40.05,h),[g(x) for x in np.arange(0,40.05,h)],color = "red")
#    plt.show()

"""

# Section 3 Question 8

gamma = 0
Omega = 1
omega = 1
a = 1
"""
deltaFileNames = ["Q3.8a.csv","Q3.8b.csv","Q3.8c.csv","Q3.8d.csv"]
for i in range(0,4):
    h = [0.3,0.3,0.3,0.049][i]
    n = int(60/h)
    delta = [0.25,0.5,1.0,20][i]
    result1 = [RK4(f,h,0,[0,0],n)[1][i][0] for i in range(0,n+1)]
    np.savetxt(deltaFileNames[i], np.transpose(([i*h for i in range(0,n+1)],result1)), delimiter = ',')
"""