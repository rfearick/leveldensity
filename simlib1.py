import numpy as np
import numpy.fft as fft
import numpy.random as rnd
#import rcnpdata as rcnp
import pywt

"""
(19/12/2017)
Copied from uct2017b/Sm154
from da2014/newlwvwlden3.py
some from da2014/newplots/SpectrumAnalyser8.py
"""

def randomWigner():
    """
    Return Wigner distributed variates
    (transformation method)
    """
    x = rnd.random()
    return np.sqrt((-4.0/np.pi)*np.log(x))  # rwf's Wigner random deviate

def randomPT():
    """
    Return Porter Thomas distributed variates
    (rejection method with intermediate transformation x->s**2)
    Range 0..10
    """
    while 1:
        s = rnd.random(2)
        if s[0] < np.exp( -100.0*s[1]*s[1]/2. ): break
    pt = 10.0*s[1]
    return pt*pt

def gauss( h, sig ):
    """
    Return Gaussian array with standard deviation sig in form useful for fft

    Parameters
    ----------
    h : ndarray of floats
         Array to be filled with function in fft ordering.
    sig :
         Standard deviation of Gaussian.
    
    Notes
    -----
    10/7/2012: fixed according to [da2011]/testconvolution.py
    """
    width = 8      # width*sig is range where gaussian is "nonzero"
    fact=1.0/(np.sqrt(2.0*np.pi)*sig)
    M=int(width*sig)
    N=len(h)
    h[0]=fact
    for i in range(1,M):
        x=float(i)
        h[i]=fact*np.exp(-x*x/(2.0*sig*sig))
        h[N-i]=h[i]


def lorentzian( h, sig ):
    """
    Return lorentzian (breit-wigner) with width sig in
    form useful for fft

    Parameters
    ----------
    h : ndarray of floats
         Array to be filled with function in fft ordering.
    sig :
         Standard deviation of Gaussian. Converted to FWHM using
         Gaussian approximation.
    
    Notes
    -----
    10/7/2012: fixed according to [da2011]/testconvolution.py
    """
    width = 15      # width*sig is range where lorentzian is "nonzero"
    sig=sig*2.3/2.0
    fact=sig/np.pi
    M=int(width*sig)
    N=len(h)
    h[0]=fact/sig/sig
    for i in range(1,M):
        x=float(i)
        h[i]=fact/(x*x+sig*sig)
        h[N-i]=h[i]

def metropolis(spect,err):
    for j in range(5):
        for i in range(len(spect)):
            trial=spect[i]+err[i]*(-0.5+random())
            if trial > 0.0 and abs(trial-spect[i]) < err[i] :
                spect[i] = trial


def convolute(spectrum, lineshape, width):
    """
    convolute spectrum with lineshape of s.dev. width
    note that width is in channels
    """
    Nspect=len(spectrum)
    g=np.zeros(Nspect*2)  
    h=np.zeros(Nspect*2)    
    g[0:Nspect]=spectrum[0:Nspect]
    g[Nspect:]=spectrum[Nspect-1::-1]
    lineshape(h,width)
    fh=fft.fft(h)
    fg=fft.fft(g)
    G=abs(fft.ifft(fh*fg))      # convoluted spectrum
    G=G[0:Nspect]               # truncate to size
    return G
 

def simulate_spectrum(Nspect, Npk, lineshape, sigma):
    """
    generate a simulted spectrum in Nspect channels
    with a mean of Npk peaks.
    Each line has a shape and width determined by linehape and sigma (in channels)
    """
    g=np.zeros(Nspect*2)
    Nspace=float(Nspect*2)/Npk      # average level spacing
    e = 0.0            # peak energy 
    for i in range(Npk):
        x=Nspace*randomWigner()          # random level space
        y=randomPT()                     # random strength
        e+=x                             # energy of level
        ne=int(e+0.5)                    # force energy to nearest channel
        if ne<Nspect: g[ne]=y                # into spectrum
    # convolute with line shape
    G=convolute(g, lineshape, sigma)*1000.0
    G=G[0:Nspect]
    return G, Nspace  # improve this ?

def Open( name, ndec=0 ):
    """
    open spectrum file and optionally decimate
    returns 2048 channel spectrum
    """
    en,a,da=np.loadtxt("../uct2011b/data/5keV/"+name, unpack=True)
    if ndec != 0:
        # add channels but keep cross section same 28/6/12
        z=np.sum(np.reshape(np.array(a),(len(a)/ndec,ndec)),1)/ndec
        #z=resize(z,(2048,))
        deltaE=(en[1]-en[0])/ndec
        en=np.sum(np.reshape(np.array(en),(len(en)/ndec,ndec)),1)/ndec
    else:
        z=a
    deltaE=en[1]-en[0]
    return (z,en,float(deltaE))

def Approximations(data, family, Nlevels):
    """
    get approximation reconstrutions at different levels
    for a DWT family.
    returns levels+1 arrays with A[0]=full reconstruction,
    and A[1]=first approx, A[levels] is smoothest
    """
    # subtract off mean data
    meandata=np.mean(data)
    #meandata=0.0
    # get DWT coefficients
    coeffs = pywt.wavedec(data-meandata, family, mode='sym',level=Nlevels)
    lcoeffs=len(coeffs)
    #print( "len coeffs",lcoeffs)
    #for i in coeffs: print( len(i),i)
    # reconstruct approximations
    A=[]
    c=pywt.waverec(coeffs,family,mode='sym')
    A.append(np.array(c)+meandata)
    for j in range(Nlevels,0,-1):
        coeffs[j][0:]=0.0
        c=pywt.waverec(coeffs,family,mode='sym')
        A.append(np.array(c)+meandata)
    return A

def Differences(approximations):
    # fixed: 30/6/2014
    l=len(approximations)
    #D=[0]*len(approximations)  ????????????????????????????
    D=[]
    D.append(np.zeros(len(approximations[0])))
    for i in range(1,l):
        D.append(approximations[i-1]-approximations[i])
    return D
        
def ApproximationsV(data, family, levels):
    """
    get approximation reconstrutions at different levels
    for a DWT family.
    returns levels+1 arrays with A[0]=full reconstruction,
    and A[1]=first approx, A[levels] is smoothest
    """
    # subtract off mean data
    meandata=np.mean(data)
    #meandata=0.0
    # get DWT coefficients
    coeffs = pywt.wavedec(data-meandata, family, mode='sym',level=Nlevels)
    lcoeffs=len(coeffs)
    for i,l in enumerate(coeffs):
        vl=np.var(l)
        l[:]=vl
        coeffs[i]=l
    #print "len coeffs",lcoeffs
    #for i in coeffs: print len(i)
    # reconstruct approximations
    A=[]
    c=pywt.waverec(coeffs,family,mode='sym')
    A.append(np.array(c)+meandata)
    for j in range(Nlevels,0,-1):
        coeffs[j][0:]=0.0
        c=pywt.waverec(coeffs,family,mode='sym')
        A.append(np.array(c)+meandata)
    return A

def Hansen(x, D, sig, alpha, ys):
    """
    Plot hansen autocorrelation formula
    x: energy offset (MeV)
    D: level density
    alpha: parameter
    ys: ratio sigmasm/sigma
    11/7/2012: fix 4->2 in last term
    """
    #ys=sigsm/sig
    ysp=1.+ys*ys
    #Act=np.zeros(len(x))
    #for i in range(Nac):
        #x=float(i)*de
    act=(1.0/(2.0*np.sqrt(np.pi)*sig))*(alpha*D)*np.exp(-x*x/(4.*sig*sig))
    act+=(1.0/(2.0*np.sqrt(np.pi)*sig*ys))*(alpha*D)*np.exp(-x*x/(4.*sig*sig*ys*ys))
    act-=(1.0/(2.0*np.sqrt(np.pi)*sig))*(alpha*D
            )*np.sqrt(8./ysp)*np.exp(-x*x/(2.*sig*sig*ysp))
        #Act[i]=act
    return act
    
def Hansen2(x, D, sig, alpha, sigsm):
    """
    Plot hansen autocorrelation formula
    x: energy offset (MeV)
    D: level density
    alpha: parameter
    ys: ratio sigmasm/sigma
    11/7/2012: fix 4->2 in last term
    """
    ys=sigsm/sig
    ysp=1.+ys*ys
    act=(1.0/(2.0*np.sqrt(np.pi)*sig))*(alpha*D)*np.exp(-x*x/(4.*sig*sig))
    act+=(1.0/(2.0*np.sqrt(np.pi)*sig*ys))*(alpha*D)*np.exp(-x*x/(4.*sig*sig*ys*ys))
    act-=(1.0/(2.0*np.sqrt(np.pi)*sig))*(alpha*D
            )*np.sqrt(8./ysp)*np.exp(-x*x/(2.*sig*sig*ysp))
        #Act[i]=act
    return act
    
def autocorrelation(Gw):
    """
    calculate autocorrelation.
    """
    fg=fft.fft(Gw-np.mean(Gw))
    Nfg=len(Gw)
    meand=np.mean(Gw)
    Accomplex=(fft.ifft(fg*np.conjugate(fg))) / float(Nfg) #- 1.0
    #Acsmooth=(fft.ifft(fh*np.conjugate(fh))) / float(Nfg) #- 1.0
    Ac=Accomplex.real/meand**2
    #Acsm=Acsmooth.real
    return Ac

