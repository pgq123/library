import numpy as np
from mpl_toolkits.mplot3d import axes3d
from matplotlib import pyplot as plt
from matplotlib import cm

y,x=np.ogrid[-2:2:200j,-2:2:200j]
z=x*np.exp(-x**2-y**2) #y.shape(200,1) x.shape(1,200)
extent=[np.min(x),np.max(y),np.min(y),np.max(y)]

fig=plt.figure(figsize=(8,6))
ax=fig.gca(projection='3d')
ax.plot_surface(x,y,z,cmap=cm.jet)
plt.show()