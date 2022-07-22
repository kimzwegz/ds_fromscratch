import pandas as pd
import sympy as sym
from sympy import *
import numpy as np
import sympy.plotting.plot as symplot
import matplotlib.pyplot as plt
import math
import mpl_toolkits
from mpl_toolkits import *
from mpl_toolkits.mplot3d import *
from matplotlib import *


x, y = symbols("x y")

fx = 3*(1-x)**2 * exp(-(x**2) - (y+1)**2) - 10*(x/5 - x**3 - y**5) * exp(-x**2-y**2) - 1/3*exp(-(x+1)**2 - y**2)

# from sympy.plotting import plot3d
# plot3d(fx, (x,-3,3), (y,-3,3)) 

fxx = lambdify([x,y], fx)
xx = np.linspace(-3,3,201)
yy = np.linspace(-3,3,201)
zz = fxx(xx,yy)


X , Y = np.meshgrid(xx,yy)

dfx_x = sym.diff(fx,x)
dfx_y = sym.diff(fx,y)


fxx = lambdify([x,y], fx)
dfxx = lambdify([x,y], dfx_x)
dfxy=lambdify([x,y], dfx_y)
xx = np.linspace(-3,3,201)
yy = np.linspace(-3,3,201)
zz = fxx(xx,yy)

X , Y = np.meshgrid(xx,yy)
Z = fxx(X,Y)
W = dfxx(X,Y)
V = dfxy(X,Y)
# plt.imshow(Z)

# fig = plt.figure(figsize=(4,4))
# ax = plt.axes(projection = '3d')
# ax.plot_surface(X,Y,Z, color = 'blue')
# # ax.plot_surface(X,Y,V)
# # ax.plot_surface(X,Y,W)
# # ax.scatter(10,10,10, color = 'black')
# plt.show()


b = np.linspace(-3,3,201)
iteration = 1000
step = 0.01
lmin = np.random.choice(b,2)
startx = lmin[0]
starty = lmin[1]
xl = []
yl = []
zl = []
pl = []
# print(lmin)
# print(startx)
# print(starty)



for i in range(iteration):
    gradx = np.array(dfxx(startx,starty))
    grady = np.array(dfxy(startx,starty))
    startx = startx - (step*gradx)
    starty = starty - (step*grady)
    point = fxx(startx, starty)
    xl.append(startx)
    yl.append(starty)
    zl.append(point)
    # print(lmin)
    # point = np.array(fxx(lmin[0], lmin[1]))
    # print(point)
# lnp = np.array(l)
# mnp = np.array(m)

# x_arr = l_arr[:,0]
# y_arr = l_arr[:,1]
# z_arr = np.array(p)

ax = plt.axes(projection = '3d')
ax.plot_surface(X,Y,Z , alpha=0.3)
ax.plot(xl, yl, zl, color = 'red', linewidth=5)
ax.scatter(xl[0], yl[0], zl[0], color = 'black')
plt.show()