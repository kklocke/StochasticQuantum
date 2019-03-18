import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

dat = np.loadtxt(sys.argv[1])

T = dat[:-1,0]
S = dat[:-1,1]
iS = dat[:-1,2]

fig, ax = plt.subplots(7,1,figsize=(14,14))
ax[0].plot(T,S)
ax[0].set_xlabel("Time")
ax[0].set_title("Mean Mag")
ax[1].plot(T,iS)
ax[1].set_xlabel("Time")
ax[1].set_title("Integrated Mean Mag")

n = len(dat[0,:])-2;
n = int(n/3)
# print(n)
leftHalf = np.zeros(len(T));
# leftHalf = np.ones(len(T))
for i in range(n):
    if i != 0:
        tmp = dat[:-1,i+3]
	ax[2].plot(T,tmp,label=str(i+1))
	itmp = np.zeros(len(tmp))
	itmp[0] = tmp[0]
	for j in range(1,len(tmp)):
	    itmp[j] = itmp[j-1] + tmp[j]
	for j in range(len(tmp)):
	    itmp[j] /= float(j+1)
	ax[4].plot(T,itmp)
    tmp2 = dat[:-1,i+3+n];
    # if i != n-1:
    #     tmp3 = dat[:-1,i+3+2*n];
    # ax[2].plot(T,tmp,label=str(i+1))
    ax[3].plot(T,tmp2,label=str(i+1))
    if i != n-1:
	tmp3 = dat[:-1,i+3+2*n]
        ax[5].plot(T,tmp3,label=str(i+2))
    	ssItmp = np.zeros(len(T))
   	ssItmp[0] = tmp3[0]
        for j in range(1,len(ssItmp)):
	    ssItmp[j] = ssItmp[j-1] + tmp3[j]
        for j in range(len(ssItmp)):
	    ssItmp[j] /= float(j+1)
        ax[6].plot(T,ssItmp,label=str(i+1));
    # if (i >= (n/2)):
	# leftHalf += tmp3
	# leftHalf *= tmp3
# leftHalf = leftHalf * np.log(leftHalf)
# ax[6].plot(T,leftHalf)
ax[2].legend(bbox_to_anchor=(1.,1.5))
ax[2].set_xlabel("Time")
ax[2].set_title("Spin Correlation")
ax[3].legend(bbox_to_anchor=(1.,1.5))
ax[3].set_xlabel("Time")
ax[3].set_title("Site Magnetization")
ax[4].set_xlabel("Time")
ax[4].set_title("Time avg Corr")
ax[5].set_title(r"$S_j$")
ax[6].set_title("Time avg EE")
ax[5].legend(bbox_to_anchor=(1.,1.5));
fig.subplots_adjust(hspace=.8)
plt.savefig(sys.argv[2],bbox_inches='tight')
plt.close('all')
