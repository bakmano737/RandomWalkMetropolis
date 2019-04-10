######################################################################
# Random Walk Metropolis #

import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

# Square Root of 2*pi
_s2p_ = 2.5066

######################################################################
# Random Walk Metropolis
#   Input Arguments:
#     F - Fitness Function
#    Fp - Fitness Function Parameter List
#    xi - Input State
#    px - Probability of input state (F(xi))
#     q - Distribution to use for random draw for step size
#    qp - Parameters for step size distribution function
#  imax - Total number of states to compute
######################################################################
def rwm(F,Fp,xi,px,q,qp,imax):
    # Iterate through the required chain length
    for i in range(imax-1):
        # Get the step from the provided distribution
        dlt = q(*qp)
        # Create the proposal point
        xip = xi[i] + dlt
        # Determine the "fitness" of the proposal point
        pxp = F(xip,*Fp)
        # Determine the Metropolis Ratio (do not divide by 0!)
        kwa = {'out':np.zeros_like(pxp),'where':px[i]!=0}
        alp = np.divide(pxp,px[i],**kwa)
        # No acceptance rates greater than 1
        acc = np.minimum(1,alp)
        # Random draw from U(0,1) for acceptance probability
        pac = np.random.rand()
        # Determine acceptance
        if acc > pac:
            # High Metropolis Ratio - accept the proposal
            xi[i+1] = xip
            px[i+1] = pxp
        else:
            #  Low Metropolis Ratio - reject the proposal
            xi[i+1] = xi[i]
            px[i+1] = px[i]
    # Return the Chain and Fitness
    return [xi,px]

######################################################################
# Standard Normal Distribution
# Expects as input a numpy array x
# Optionally takes arguments m (mu) and d (sigma)
# m (mu) represents mean and d (sigma) standard deviation
######################################################################
def snd(x,m=0,d=1):
    return (1/(d*_s2p_))*np.exp(-0.5*((x-m)/d)**2)
######################################################################

######################################################################
def sndTest(m=0,d=1):
    x = np.arange(-5.0,5.0,0.01)
    lbl = "Mean:{0:1.2f}, Std:{1:1.2f}".format(m,d)
    plt.plot(x,snd(x,m,d),label=lbl)
    plt.ylabel("p(x|m,d) - Normal Distribution")
    plt.xlabel("x")
    plt.legend()
    plt.show()
######################################################################

######################################################################
def sndrwm():
    # Allocate Variables
    T  = 50000
    xi = np.empty(T)
    pr = np.empty(T)
    # Chain State Initialization
    xi[0] = np.array([10*rnd.rand()-5])
    pr[0] = np.array(snd(xi[0]))
    # Random Walk Metropolis
    xo,po = rwm(snd,[-2.0,0.1],xi,pr,rnd.uniform,[-0.5,0.5],T)
    # Remove Burn-in
    xon = xo[5000:]
    pon = po[5000:]
    print(xon)
    print(pon)
    plt.plot(xon,pon,'go',label="Sim PDF",zorder=0)
    # Define histogram bin edges
    bnw = 0.10
    bns = np.arange(-5.0,5.0,bnw)
    # Histogram Plot Keyword Arguments
    kwa = {'density':'True','bins':bns,'label':'Hist','zorder':10}
    # Create and Plot the histogram
    hist,bins,ptch = plt.hist(xon,**kwa)
    # Create the cumulative distribution
    cdf = np.cumsum(hist*bnw)
    # Determine the 95%CI edges from CDF
    ci25  = np.interp(0.025,cdf,bins[:-1])
    ci975 = np.interp(0.975,cdf,bins[:-1])
    # Compute and Print Mean, Standard Deviation and 95CI
    print(np.mean(xon))
    print(np.std(xon))
    print(ci25, ci975)
    # Plot the simulated PDF and calculated CDF
    plt.plot(bins[:-1],cdf,'r', label="CDF")
    # Finally, axis labels and legend
    lbl="Normalized Frequency (%)"
    plt.ylabel(lbl)
    plt.xlabel("x")
    plt.legend()
    plt.show()
######################################################################

######################################################################
# Bimodal Normal Distribution
# Expects as input a numpy array x
# Optionally takes means and std.devs for each mode (m1/2,d1/2) 
def bnd(x,m1=-5,d1=1,m2=5,d2=1):
    return snd(x,m=m1,d=d1)/3 + 2*snd(x,m=m2,d=d2)/3

def bndTest():
    x = np.arange(-9.0,9.0,0.01)
    bndx = bnd(x)
    print(np.mean(bndx))
    print(np.std(bndx))
    plt.plot(x,bndx)
    plt.show()

def bndRWM():
    # Allocate Variables
    T  = 25000
    xi = np.empty(T)
    pr = np.empty(T)
    # Chain State Initialization
    xi[0] = np.array([20*rnd.rand()-10])
    pr[0] = np.array(snd(xi[0]))
    # Random Walk Metropolis
    xo,po = rwm(bnd,[],xi,pr,rnd.uniform,[-0.5,0.5],T)
    # Remove Burn-in
    xon = xo[15000:]
    pon = po[15000:]
    plt.plot(xon,pon,c='g',label="Sim PDF")
    # Define histogram bin edges
    bns = np.arange(-9.0,9.0,0.5)
    # Histogram Plot Keyword Arguments
    kwa = {'density':'True','bins':bns,'label':'Hist'}
    # Create and Plot the histogram
    hist,bins,ptch = plt.hist(xon,**kwa)
    # Create the cumulative distribution
    cdf = np.cumsum(hist*0.5)
    # Determine the 95%CI edges from CDF
    ci25  = np.interp(0.025,cdf,bins[:-1])
    ci975 = np.interp(0.975,cdf,bins[:-1])
    # Compute and Print Mean, Standard Deviation and 95CI
    print(np.mean(xon))
    print(np.std(xon))
    print(ci25, ci975)
    # Plot the simulated PDF and calculated CDF
    #plt.plot(xon,      pon,c='g',label="Sim PDF")
    plt.plot(bins[:-1],cdf,c='r',label="CDF",markevery=[ci25,ci975])
    # Finally, axis labels and legend
    lbl="Normalized Frequency (%)"
    plt.ylabel(lbl)
    plt.xlabel("x")
    plt.legend()
    plt.show()

#sndTest()
sndrwm()
#bndTest()
#bndRWM()