import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

dat = np.loadtxt(sys.argv[1])

T = dat[:-1,0]
S = dat[:-1,1]
iS = dat[:-1,2]

fig, ax = plt.subplots(4,1,figsize=(6,6))
ax[0].plot(T,S)
ax[0].set_xlabel("Time")
ax[0].set_title("Mean Mag")
ax[1].plot(T,iS)
ax[1].set_xlabel("Time")
ax[1].set_title("Integrated Mean Mag")

n = len(dat[0,:])-3;
n = int(n/2)
# print(n)
for i in range(n):
    if i != 0:
        tmp = dat[:-1,i+3]
	ax[2].plot(T,tmp,label=str(i+1))
    tmp2 = dat[:-1,i+3+n];
    # ax[2].plot(T,tmp,label=str(i+1))
    ax[3].plot(T,tmp2,label=str(i+1))
ax[2].legend(bbox_to_anchor=(1.,1.5))
ax[2].set_xlabel("Time")
ax[2].set_title("Spin Correlation")
ax[3].legend()
ax[3].set_xlabel("Time")
ax[3].set_title("Site Magnetization")
fig.subplots_adjust(hspace=.8)
plt.savefig(sys.argv[2],bbox_inches='tight')
plt.close('all')
