from __future__ import print_function
import pylab as plt
import numpy as np
import numpy.fft as fft
import numpy.random as rnd
import scipy.io as sio
import scipy.signal as signal
import os
import sys
from scipy.optimize import curve_fit
from scipy.special import erfc

import simlib1 as sim
import leveldensities as lden
from leveldensityanalysisplots import *
import spectrumtools as tools

##plt.rc('text',usetex=True)
##plt.rc('mathtext',default='rm')
##plt.rc('font', family='serif', size=12.0)
##plt.rc('figure.subplot',top=0.94, right=0.95)
##plt.rc('legend',fontsize='small')
##plt.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})

plt.rc('font',size=10)

# shorthand function calls
pi=np.pi
sqrt=np.sqrt
cos=np.cos
sin=np.sin
exp=np.exp

versiondate="19/12/2017"
versionname="leveldensitywithnoise.py"

#rnd.seed( 1213141516 )  # fix state of RNG

"""
(19/12/2017)
Copied from uct2017b/Sm154
(21/8/2017)
from noisylevel_2.py
- remove emd stuff
- remove linear fit to widths
- KISS 
(20/8/2017)
noisylevel_2.py
(18/8/2017)
from emdlevel2017_12.py
(2/8/2017)
from da2017/Sm154/emdlevel2017_2.py
(28/7/2017)

"""
#
# Some parameters
#

nucleus="144Sm"
nucleus="154Sm"
#nucleus="144Nd"
A=int(nucleus[0:3])
Z=62
N=A-Z

# set up AC windows.
# divide ROI into NAC bins.
# do AC in bins offset by width(bin)/NoffsetAC . default 4, 2
NAC=4        # No. of AC bins
NoffsetAC=1  # No. of offsets per AC window
NACslides=5 # no. of AC window slides per AC bin

print("Level density analysis for", nucleus)
print("A=",A,", Z=",Z,", N=",N)

"""
OK, we now read in a spectrum and fluctuation analysis
"""

from pathlib import Path
def findstore(subdir):
    p=Path.cwd()
    partlist=list(p.parts)
    #print(partlist)
    if "linax" in partlist:
        i=partlist.index("linax")
        pf=Path(*partlist[0:i+1])
        pf=pf /"datastore"/subdir
    else:
        print("path problem")
        pf=Path(".")
    #print("path:",pf)
    return pf.as_posix()+"/"

xnucleus=nucleus[3:]+nucleus[0:3]  # fiddle name and A !
path=findstore(xnucleus)
#print(path)

# NOTE:   below: AC => autocorrelation

# some useful functions
# --- report M1 fraction at energy e
em1=None
m1fraction=None
m1lo=None
m1hi=None
def M1(e):
    if e<=m1lo or e>=m1hi:
        return 0.0
    for i in range(len(em1)):
        if em1[i]-0.200<e<em1[i]+0.200:
            return m1fraction[i]
    return 0.0 # default for fails

def Csq(x,sig,a):
    X=x/(np.sqrt(2.0)*sig)
    A=(x+a)/(np.sqrt(2.0)*sig)
    Am=(x-a)/(np.sqrt(2.0)*sig)
    A=x/(np.sqrt(2.0)*sig)+a
    Am=x/(np.sqrt(2.0)*sig)-a
    csq=erfc(Am)-erfc(A)
    return (erfc(Am)-erfc(A))/csq[0]


# --- calculate alphaIg: we will assume N1=N2 so beta=1/alpha
def alphaIg(alpha, beta):
    """
    alpha=<I2>/<I1>
    beta=N1<I1>/N2<I2>
    """
    return 3*(1+beta/alpha)*(1+alpha*beta)/(1+beta)**2 -1

def Hansen2(x, D, sig, alpha, sigsm):
    """
    Plot hansen autocorrelation formula
    x: energy offset (MeV)
    D: level density
    alpha: parameter
    ys: ratio sigmasm/sigma
    11/7/2012: fix 4->2 in last term
    """
    global sigsmn, sigsm0
    #sige=np.sqrt(sig**2-sigsmn**2)
    #sigw=np.sqrt(sige**2+sigsm0**2)
    #ys=sigw/sig
    ys=sigsm/sig
    ysp=1.+ys*ys
    act=(1.0/(2.0*np.sqrt(np.pi)*sig))*(alpha*D)*np.exp(-x*x/(4.*sig*sig))
    act+=(1.0/(2.0*np.sqrt(np.pi)*sig*ys))*(alpha*D)*np.exp(-x*x/(4.*sig*sig*ys*ys))
    act-=(1.0/(2.0*np.sqrt(np.pi)*sig))*(alpha*D
            )*np.sqrt(8./ysp)*np.exp(-x*x/(2.*sig*sig*ysp))
        #Act[i]=act
    return act

def actheory(x,sign,sigw):
    """
    Plot hansen autocorrelation formula
    x: energy offset (MeV)
    sign: narrow smooth
    sigw: wide smooth
    """
    ys=sigw/sign
    #print(ys)
    ysp=1.+ys*ys
    act=(1.0/sign)*np.exp(-x*x/(4.*sign*sign))
    act+=(1.0/(sign*ys))*np.exp(-x*x/(4.*sign*sign*ys*ys))
    act-=(1.0/sign)*np.sqrt(8./ysp)*np.exp(-x*x/(2.*sign*sign*ysp))
    return act


# define fitting functions with 1 or 2 parameters
def fitf(Xac,D):
    global sig,alpha,ys
    return sim.Hansen(Xac, D,sig,alpha,ys)
def fitf2(Xac,D,sig):
    global alpha,sigsm,sigx,initialD
    return Hansen2(Xac, D*initialD,sig,alpha,sigsm)*np.exp(-Xac**2/(2.0*sigx**2))

def fitf3(Xac,D,sig, sigsm):
    global alpha
    return sim.Hansen2(Xac, D,sig,alpha,sigsm)

# find zeros in numpy array
def findzeros(array):
    """
    very crude: only for smoothly varying stuff like autocorrelations
    """
    zerolist=[]
    for i,x in enumerate(array[1:]):  # note i=1 means index 1
        #print(i,x)
        if x*array[i]<0.0:  # (i-1)+1 !
            #print(i,x,array[i])
            zerolist.append(i+1)
    return zerolist


# ----------------------------------------------------------

# The following parameters should probably remain fixed

# lineshape for convolutions
lineshape=sim.gauss

# do we use narrow smoothing as well as wide?
NarrowSmooth=False

# set smoothing parameters in channels
smoothwide=3.5 # from 3.5
smoothnarrow=0.39*2.35

alpha=2.0+0.273            # default for N=1

# fit 1 or 2 parameters in fits to autocorrelation (AC) function
fit2params=True

# ---------------------------------------------------------

# this is a loop inherited from s-d shell analysis; keep it until we're sure we don't need it
for targetname in [nucleus]:
    firstpass=True
    targetbase=targetname[0:5]
    
    # take sig to be what Andi thinks -- this does need to be experimental sig
    # -- note: can't rely on AC function to produce correct value !! See notes.
    esig,sig=0.0,50.0 # fwhm of experimental resolution in keV  <<<<<<<<<<<<<<<<<<<<
    print("Assumed fwhm of resolution fn is %6.3f keV"%(sig))
    print("Smooth factors: wide %6.3f; narrow %6.3f"%(smoothwide,smoothnarrow))
    # convert to sdev of gaussian in MeV
    sig/=1000.0   # sig in MeV
    sig/=2.35      # FWHM to stdev
    #print("sig",sig, sig*2.3)
    # calculate smoothing widths
    sigsmn=smoothnarrow*sig       # < sig   . should be 0.5*FWHM ?
    sigsm=smoothwide*sig       # in MeV
    sigexpt=sig
    sigsm0=sigsm
    sigx=sigsm0*2.5
    print("Calculated: sig wide %6.3f; sig narrow %6.3f"%(sigsm,sigsmn))
    DefineParameters(sig,sigsmn if NarrowSmooth else 0.0,sigsm)

    # we have only one angle ...
    for targetangle in ["00"]: 
        """
        Iterate over all targets at zero degrees
        """
        angle=0.4
        lump=1
        if nucleus=="144Sm":
            targetfilename="cs_full_5kev.dat"
            #targetfilename="cs_full_3deg.dat"
        elif nucleus=="154Sm":
            targetfilename="data_154Sm_all_5keV.dat"
        elif nucleus=="142Nd":
            targetfilename="datafile-142Nd-hExdiff-10keV.dat"
        elif nucleus=="144Nd":
            targetfilename="datafile-144Nd-hExdiff-10keV.dat"

        # *** set up ROIs : Note that spectrum file may not start from 0.0 !!
        # at 25 keV / channel
        ROIlo,ROIhi=(40,340)      # 300 chans
        # at 5 keV / channel
        ROIlo,ROIhi=(40*5,360*5)  # 1600 chans: 200 - 1800
        # at 10 kev?ch
        #ROIlo,ROIhi=(40*5//2,360*5//2) # 750 chans
        #ROIlo,ROIhi=(900,1500)
        
        ROIlo=ROIlo+50
        ROIhi=ROIhi+50

        # *** Read in data
        try:
            print("data input from ",path+targetfilename)
            e,cs0=np.loadtxt(path+targetfilename,unpack=True,usecols=(0,1))
            #e,cs0=np.loadtxt(path+"../Sm154/"+"data_154Sm_all_5keV.dat",unpack=True,usecols=(0,1))
            ##e,cs0=np.loadtxt(path+"cs_full_3deg.dat",unpack=True,usecols=(0,1))
            de=e[2]-e[1]
            # energy in MeV: find kev/ch as int
            intde=int(de*1000.0+0.1)
            G=cs0
            ##############G=0.02*rnd.randn(len(cs0))+5.0    # (simulate for testing)
            X=e
            if nucleus=="154Sm":
                M1fractions=True
                # get M1 fractions
                em1,m1fraction=np.loadtxt("../../da2016/Sm154/M1fraction.dat",unpack=True)
                # these are in bins of width 0.4 MeV: get bin limits
                m1lo=em1[0]-0.2
                m1hi=em1[-1]+0.2
            elif nucleus=="144Sm":
                M1fractions=False
                E0=7.63
                gamma=2.0
                m1fraction=0.25*(gamma*E0)*(e*gamma/((e**2-E0**2)**2+(e*gamma)**2))
            else:
                M1fractions=False
        except:
            print("File not found",targetfilename)
            sys.exit(3)
        

        Nspect=len(G)
        Nlo=ROIlo
        Nhi=ROIhi
        Exlo,Exhi=(e[ROIlo],e[ROIhi])
        print("Region of interest ",Exlo,' to ',Exhi, ' MeV')

        #print("ROI for analysis",targetbase, Nlo, Nhi, ROIlo, ROIhi)

        # *** background subtraction
        #THERE IS NO BACKGROUND 
        bkg=0.0
        """
        if NarrowSmooth:
            Gn=sim.convolute(G, lineshape, sigsmn/de)
            iAC=0             # first point in autocorrelation fit
            hansen0=0.0       # energy offset of first point
        else:                 # if no narrow smooth, ignore noise in first channel
            Gn=G
            iAC=1
            hansen0=de

        
        # smooth spectrum by convoluting with gaussian of width sigsm channels: g_>
        Gs=sim.convolute(G, lineshape, sigsm/de)
        Gs[Gs<0.2]=0.2  #  lower limit (prevents div by zeros) needs more thought ...
        """
        # increase sig to account for convolution of g
        # note that the actual value depends on sig!
        sigw=np.sqrt(sig**2+sigsm**2)#sqrt(1.25)*sig
        sign=sig
        if NarrowSmooth:
            sign=np.sqrt(sig**2+sigsmn**2)#sqrt(1.25)*sig
        ys=sigw/sig
        print("fluctuation parameters sigsm:%6.3f, sig:%6.3f, sigsmn:%6.3f, ys:%6.2f"%(sigsm,sig,sigsmn,ys))
        print("fluctuation parameters fwhm. sigsm: %6.3f, sig: %6.3f, sigsmn: %6.3f"%(sigsm*2.3,sig*2.3, sigsmn*2.3))
        """
        # make ratio spectrum in full ROI
        widthROI=ROIhi-ROIlo
        TG=G[ROIlo:ROIhi]
        TGn=Gn[ROIlo:ROIhi]
        TS=Gs[ROIlo:ROIhi]
        Gratio0=G[ROIlo:ROIhi]/Gs[ROIlo:ROIhi]
        Gratio=Gn[ROIlo:ROIhi]/Gs[ROIlo:ROIhi]
        Xratio=X[ROIlo:ROIhi]
        """
        widthROI=ROIhi-ROIlo
        S=tools.RatioSpectrum( X, G, targetfilename, ROIlo, ROIhi )
        S.makeRatioSpectra( sigsmn*1000.0, sigsm*1000.0 )
        if NarrowSmooth:
            Gratio=S.rationar
            iAC=0
            hansen0=0.0
        else:
            Gratio=S.ratioraw
            iAC=1
            hansen0=de
        Xratio=S.Eratio
        Gs=S.gwide
        Gratio0=S.ratioraw
        TG=S.gROI
        TS=S.gROIw
        TGn=S.gROIn
        
        # find basic details from full autocorrelation
        allAC=sim.autocorrelation(Gratio)
        #Grxx=Gratio-np.mean(Gratio)
        #allAC2=np.correlate(Grxx,Grxx,mode="same")[0:len(Gratio)]/len(Gratio)#-1.0
        #print("AC***",np.shape(allAC),np.shape(Gratio))
        widthROId2=widthROI//2
        zerolist=findzeros(allAC)
        widthROId2=zerolist[1]+1
        print("allAC fit2 input ",alpha,ys,widthROId2)
        ###ys=3.9716 ####
        initialD=allAC[iAC]/sim.Hansen(hansen0,1.0,sig,alpha, ys)
        fitX=Xratio[iAC:widthROId2]-Xratio[0]
        fitallAC=allAC[iAC:widthROId2]
        initial=1.0
        popt,pcov=curve_fit(fitf2, fitX, fitallAC, p0=[initial,sig])
        print("Derived resolution sigold, signew, fwhm is %f6.3, %6.3f, %6.3f"%(sig,popt[1], popt[1]*2.3))
        # plot AC of full ROI
        PlotFullAC(11, Xratio,allAC,popt,alpha,sig,sigsm,initialD,A)
        
        # update sig
        sig=popt[1]
        print("sig reset to ",sig, " after fit to full region")
        #sig=0.040/2.3 # force a value  <<<<<<<<<<<<<<<<<<<<<<<<<

                # plots: 1 ----------------------------
        PlotSpectrum(1,X,G,Gs,ROIlo,ROIhi,A,incounts=True)

        #print(pcov,popt)
        #plt.show()
        #sys.exit(3)

        # setup for calculation of autocorrelation by Wiener-Khinchin in autocorrelation windows

        # calculate ac windows
        lohi=[]
        lenGr=len(Gratio)
        dAC=len(Gratio)//NAC
        AChi=dAC
        AClo=0
        offsetAC=dAC//NoffsetAC
        while AChi<=lenGr:
            lohi.append([AClo,AChi])
            AClo+=offsetAC
            AChi+=offsetAC
        NACbins=len(lohi)  # no. of autocorrelation bins
        print("AC bin sizes",lenGr,dAC,offsetAC)

        # prepare to plot autocorrelations in loop
        # save:
        Ed=[]  # energy of leveldensity point
        Dd=[]  # level density
        Ddh=[] # level density upper uncertainty
        Ddl=[] # level density lower uncertainty
        Dr=[]  # level density calculated from raw autocorrelation, no fit.
        Sigfit=[] # fitted sig
        ACzero=[] # AC at zero offset
        Alpha=[]
        varn=[]
        ##sig=0.035 # !!!!!!!!!!!!!!!!!!!
        sig0=sig  # keep an initial value for fits
        sig0=0.025
        for i,lh in enumerate(lohi[:-1]):
            for slide in range(0,dAC,dAC//NACslides):  # implement sliding window
                sig=sig0
                #sig0+=0.001
                # loop over windows to get AC and LD
                AClo,AChi=lh
                AClo+=slide
                AChi+=slide
                # calculate autocorrelation in current bin
                Gw=Gratio[AClo:AChi]
                Gw0=Gratio0[AClo:AChi]
                if slide==0:
                    print()
                    print("AC no.:",i,AClo,AChi,AChi-AClo,len(Gw)," range(MeV)",X[ROIlo+AClo],X[ROIlo+AChi])
                else:
                    print()
                if len(Gw) != AChi-AClo: break
                Ac=sim.autocorrelation(Gw)
                # correct for noise
                Ac0=sim.autocorrelation(Gw0)
                meanG=np.mean(TG[AClo:AChi])
                varGw0=np.var(Gw0-1.0)
                varnoise0=(Ac0[0]-Ac0[1])
                signoise=np.sqrt(sigsmn**2+de**2)
                signoise=sigsmn
                f1=2.0*np.sqrt(np.pi)*(signoise/de) 
                varnoise=varnoise0/f1
                print("noise %8.5f %8.5f %8.5f %8.5f %8.5f"%(varGw0,varnoise0,varnoise, Ac0[0], Ac[0]))
                if varnoise>Ac[0]: continue
                #print("Noise",Ac[0],varnoise,Ac0[0],Ac0[1],f1,f1*meanG**2)
                # 
                meanE=(X[ROIlo+AClo]+X[ROIlo+AChi-1])/2.0
                meanE=(Xratio[AClo]+Xratio[AChi-1])/2.0  # mean energy of bin
                meanE0=meanE # keep copy of mean bin energy

                # ******************************
                # estimate what mean energy should be according to level densities
                Exx=plt.linspace(Xratio[AClo],Xratio[AChi-1], 200)
                # init level density theory
                #Z,N=62,144-62
                Spin=1.0
                rhoxx=lden.LevelDenRauscher(Exx, Z, N, Spin)
                meanxx=np.sum(rhoxx*Exx)/np.sum(rhoxx)
                print("Mean E: as calc:",meanE," from ld:",meanxx," range",Xratio[AClo],Xratio[AChi-1])
                # reset mean energy
                meanE=meanxx

                # precompute for Hansen formula
                ys=sigw/sign
                #alpha=2.0+0.273           
                # calculate alpha from M1 fraction assuming N1=N2
                # first get mean m1 fraction
                if M1fractions:
                    if nucleus=="154Sm":
                        fraction=np.zeros(len(Gw))
                        for j in range(len(fraction)):
                            ej=Xratio[AClo+j]
                            fraction[j]=M1(ej)
                    elif nucleus=="144Sm":
                        fraction=m1fraction[ROIlo+AClo:ROIlo+AChi]
                    meanM1=np.mean(fraction)
                    a=meanM1/(1.0-meanM1)
                    #print("M1 contrib",meanE0,meanM1,a)
                    if a<0.01:
                        alpha=2.27
                    else:
                        b=1.0/a
                        aIg=alphaIg(a,b)
                        aIg2=3.0*meanM1**2*(1.0+(1.0/meanM1-1)**2)*2-1  # from Kilgus: gives same result! 
                        print( "M1 fraction",meanM1," alphaIg:",aIg,aIg2," ys:",ys)
                        alpha=aIg+0.52  # add alphaD
                    #if meanM1<0.15: alpha=2.27
                else:
                    alpha=2.27
                ####alpha=alpha+1
                # fit hansen formula to data
                Nac=dAC//2            # meaningful range of AC is 0.5 bin
                if Nac > (AChi-AClo): # should not ever be true !! Should kill and fix.
                    Nac=AChi-AClo
                    if slide==0:print("*** reset Nac **")
                #print Nac, AChi, AClo

                # calculate level density by fit  ------
                initialD=Ac[iAC]/sim.Hansen(hansen0,1.0,sig,alpha, ys)
                Nacf=(AChi-AClo)//2
                zerolist=findzeros(Ac)
                #if slide==0:print("zerolist",zerolist)
                #Nacf=(zerolist[1]+5)//2 # add a bit ...
                #Xac=Xratio[AClo+iAC:AClo+Nacf]-Xratio[AClo+iAC]
                Xac=Xratio[AClo+iAC:AClo+Nacf]-Xratio[AClo]
                ACf=Ac[iAC:Nacf]
                #noisecorrection=varnoise*Csq(Xac,signoise*2.0,de/2.)
                noisecorrection=varnoise*signoise*actheory(Xac,signoise,sigsm0)#np.exp(-Xac**2/(4.0*signoise**2))
                if NarrowSmooth: ACf=ACf-noisecorrection
                print("fit 2 in: sig, alpha, ys, Nacf %6.3f %6.3f %6.3f %d"%(sig,alpha,ys, Nacf))
                print("fit 2 in: sigsm, sigsm0 %6.3f %6.3f"%(sigsm,sigsm0))
                if fit2params:
                    uncert=np.linspace(ACf[0]/10,ACf[0]/5,Nacf)
                    ACfx=ACf*np.exp(-Xac**2/(2.0*sigx**2))
                    initial=1.0
                    popt,pcov=curve_fit(fitf2, Xac, ACf, p0=[initial,sig])
                    sig=popt[1]
                    popt[0]*=initialD
                    sige=np.sqrt(sig**2-sigsmn**2)
                    sigw=np.sqrt(sige**2+sigsm0**2)
                    #print("sigs",sig,sige,sigsm,sigw)
                    ys=sigw/sig
                    sigsm=sigw
                    print("fit 2 out: sige, sig, ys %6.3f %6.3f %6.2f"%(sige,sig,ys))
                    if sige<sigsmn: continue
                else:
                    popt,pcov=curve_fit(fitf, Xac, ACf, p0=[initialD])
                # breakout 
                # ----------------------
                fitD=popt[0]#*(sig/sigaim)

                # first pass at D
                fitD0=Ac[iAC]/sim.Hansen(hansen0,1.0,sig,alpha, ys)
                # now correct for noise
                #fitD=(Ac[iAC]-varnoise)/sim.Hansen(hansen0,1.0,sig,alpha, ys)
                varn.append(varnoise)
                # D is level spacing of all multipolarities; must reduce for just one
                Dlev=fitD      # keep copy for plot
                print("fitted D",fitD0,fitD)
                if np.isnan(fitD) or np.isnan(fitD0): continue
                Ntypes=1
                if M1fractions:
                    Ntypes=2
                fitD=fitD*Ntypes  # compensate for two sets of levels: spacing = fitD*(N1+N2)/N1

                #calculate level density using raw autocorrelation ---
                #meanD=Ac[iAC]/sim.Hansen(hansen0,1.0,sig0,alpha, sigsm/sig0)
                meanD=Ac[iAC]/sim.Hansen(hansen0,1.0,sig,alpha, ys)
                meanD=meanD*Ntypes # compensate for 2 sets of levels.

                # determine uncertainties in results
                # method 1 for uncertainties: from covar matrix of fit
                Derr=np.sqrt(pcov[0,0])
                #print("fit",fitD,meanD, " AC[0]",
                #      sim.Hansen(0.0,fitD,sig,alpha,ys),
                #      1.0/sim.Hansen(0.0,1.0,sig,alpha,ys))#,popt[1])
                meanDh=fitD-Derr*Ntypes # compensate for 2 sets of levels.
                meanDl=fitD+Derr*Ntypes
                levelden=1.0/fitD    # prop. to alpha!
                #levelden=1.0/meanD
                leveldenh=1.0/meanDh-levelden
                leveldenl=levelden-1.0/meanDl
                if fit2params:   # include error on sig
                    sigerr=np.sqrt(pcov[1,1])/popt[1] # relative err.
                    leveldenh=np.sqrt(leveldenh**2/levelden**2+sigerr**2)*levelden
                    leveldenl=np.sqrt(leveldenl**2/levelden**2+sigerr**2)*levelden
                # method 2 for uncertainties: from sdev of ac
                isig=int(3*np.rint(sig/de)+0.0001)  # sig in channels
                ##std=np.std(Ac[iAC:Nac])
                ##std=np.std(Ac[isig:Nac-isig])
                std=np.std(Ac[10:Nac-10])
                ld2h=sim.Hansen(0.0,1.0,sig,alpha, ys)/(Ac[iAC]-std)/Ntypes-1.0/meanD
                ld2l=1.0/meanD-sim.Hansen(0.0,1.0,sig,alpha, ys)/(Ac[iAC]+std)/Ntypes
                if slide==0:
                    h=sim.Hansen(0.0,1.0,sig,alpha, ys)
                    #print("levden",1.0/meanD, h, Ac[iAC],std,h/Ac[iAC]/Ntypes,h/(Ac[iAC]-std)/2,h/(Ac[iAC]+std)/2)
                    #print("levden", levelden, (leveldenh-leveldenl)/2, (ld2h-ld2l)/2,leveldenh,leveldenl)

                # now set up to plot
                # Hansen formula
                Act=sim.Hansen(X[0:Nac]-X[0], Dlev, sig, alpha, ys)
                # plot autocorrelation and save Hansen result
                # if outside 95% interval of random fluctuations.
                # (important if no narrow smooth)
                validD=True
                if Ac[iAC]>1.5*std:  # was 2, now 3.
                    Ed.append(meanE)
                    Dd.append(levelden)
                    Dr.append(1.0/meanD)
                    Sigfit.append(sig)
                    #leveldenh=0.0
                    #leveldenl=0.0
                    #ld2h=0.0
                    #ld2l=0.0
                    Ddh.append(np.sqrt(leveldenh**2+ld2h**2))
                    Ddl.append(np.sqrt(leveldenl**2+ld2l**2))
                    ACzero.append(Ac[iAC])
                    Alpha.append(alpha)
                else:
                    validD=False

                print("AC**",i+(NAC-1)*slide//(dAC//NACslides),i,slide,NACslides,"alpha=%5.2f"%(alpha,))
                #if i+slide*NACslides>NACbins*2: continue
                #if slide>0: continue    # only plot for first pass
                #print("** i",i)
                #print((NAC-1)*slide//(dAC//NACslides),(NAC-1)*NACslides)
                if validD:
                    # plot autocorrelation results as we go ...
                    #Act[iAC:Nacf]+=noisecorrection
                    PlotACs(100, i+(NAC-1)*slide//(dAC//NACslides), X,Ac, Act, NAC*NACslides, Nac, meanE, targetname, NarrowSmooth)
                    plt.plot(Xac,ACf,'m-',drawstyle='steps-mid')
                    plt.plot(Xac,np.exp(-Xac**2/(2.0*sigx**2))*ACf[0])
                    if NarrowSmooth:
                        Act[iAC:Nacf]+=noisecorrection
                        plt.plot(X[0:Nac]-X[0],Act,'g-')
                    plt.text(0.5,0.8,r"sig %5.3f"%sig,fontsize=10,transform=plt.gca().transAxes)


                
        #plt.savefig(os.path.join(outputdirname,"%s%sfig2.png"%(targetbase,targetangle)))
        Dd=plt.array(Dd)
        Ddl=plt.array(Ddl)
        Ddh=plt.array(Ddh)
        
# now plot level densities,  expt and theory as fn. of E
Ex=plt.linspace(4.5,11.5, 200) # for level density calcs.
PlotLevelDensities(5,N,Z,A,Ed,Dd,Ddl,Ddh,Ex,Exlo,Exhi)

# now plot summary plots
Tn=TGn if NarrowSmooth else None
PlotAnalysisROI(8, Xratio, TG, TS, Tn, A)
#PlotStationarySpectrum(9, Xratio, TGn, TS, A)
PlotStationarySpectrum(9, Xratio, Gratio, None, A)


plt.show()
