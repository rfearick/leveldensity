"""
Support classes for spectrum analysis

(LD: level density)

Spectrum:        hold basic data for spectrum read in from file.
RatioSpectrum:   generate ration of smoothed spectrum for LD analysis.
"""

import numpy as np
import simlib1 as sim
from pathlib import Path


# find zeros in numpy array
def findzeros(energy, data):
    """
    very crude: only for smoothly varying stuff like autocorrelations
    """
    de=energy[1]-energy[0]
    zerolist=[]
    zeroenergy=[]
    for i,x in enumerate(data[1:]):  # note i=1 means index 1
        #print(i,x)
        if x*data[i]<0.0:  # (i-1)+1 !
            #print(i,x,array[i])
            zerolist.append(i+1)
            e0 = energy[i]-de*x/(data[i]-x) #linear interpolation
            zeroenergy.append(e0)
    return zerolist,zeroenergy

def findminmax(energy, data):
    """
    very crude: only for smoothly varying stuff like autocorrelations
    """
    de=energy[1]-energy[0]
    minlist=[]
    maxlist=[]
    minenergy=[]
    maxenergy=[]
    ddata=np.zeros(len(data))
    for i,x in enumerate(data[1:-1]):  # note i=1 means index 1
        ddata[i]=(data[i+1]-data[i-1])/2.0  # numerical deriv
        #print(i,x)
    for i,x in enumerate(ddata[1:-1]):  # note i=1 means index 1
        if ddata[i-1]*ddata[i+1]<0.0:  # (i-1)+1 !
            #print(i,x,array[i])
            #zerolist.append(i)
            slope=ddata[i+1]-ddata[i-1]
            e0 = energy[i-1]-2.0*de*ddata[i-1]/(ddata[i+1]-ddata[i-1]) #linear interpolation
            # only accept min if data negative, max if data positive 
            if slope > 0.0 and data[i]<0.0:
                minenergy.append(e0)
                minlist.append(data[i])
            elif slope < 0.0 and data[i]>0.0:
                maxenergy.append(e0)
                maxlist.append(data[i])
    return minenergy,maxenergy,minlist,maxlist

def AC_shape( e, sign, sigw ):
    """
    Theoretical function for autocorrelation.
    e: energy offset (MeV)
    sign: narrow smooth
    sigw: wide smooth
    """
    ys=sigw/sign
    ysp=1.+ys*ys
    act=(np.exp(-e*e/(4.*sign*sign))+
         np.exp(-e*e/(4.*sign*sign*ys*ys))/ys-
         np.sqrt(8./ysp)*np.exp(-e*e/(2.*sign*sign*ysp)))/sign
    #act=(1.0/sign)*np.exp(-x*x/(4.*sign*sign))
    #act+=(1.0/(sign*ys))*np.exp(-x*x/(4.*sign*sign*ys*ys))
    #act-=(1.0/sign)*np.sqrt(8./ysp)*np.exp(-x*x/(2.*sign*sign*ysp))
    return act

def ACtheory(x, sign, sigw, alpha, D):
    """
    'Hansen' autocorrelation formula for spectrum
    x: energy offset (MeV)
    sign: narrow smooth or sigexpt, >= sigexpt
    sigw: wide smooth
    alpha: variance parameter
    D: level spacing
    11/7/2012: fix 4->2 in last term
    """
    sigtsq=sign**2+sigw**2
    A=(alpha*D)/(2.0*np.sqrt(np.pi))
    act=A*( np.exp(-x*x/(4.*sign*sign))/sign+
            np.exp(-x*x/(4.*sigw*sigw))/sigw-
            np.sqrt(8./sigtsq)*np.exp(-x*x/(2.*sigtsq))
            )
    return act

def ACnoise(x, sign, sigw, dE, varn):
    """
    'Hansen' autocorrelation formula for noise
    x: energy offset (MeV)
    sign: narrow smooth or sigexpt, >= sigexpt
    sigw: wide smooth
    dE: level densityspectrum bin width
    varn: relative noise variance parameter
    """
    sigtsq=sign**2+sigw**2
    A=(dE*varn)/(2.0*np.sqrt(np.pi))
    act=A*( np.exp(-x*x/(4.*sign*sign))/sign+
            np.exp(-x*x/(4.*sigw*sigw))/sigw-
            np.sqrt(8./sigtsq)*np.exp(-x*x/(2.*sigtsq))
            )
    return act
    
def ACnoise_nonsm(x, sign, sigw, dE, varn):
    """
    'Hansen' autocorrelation formula for noise with NO narrow smooth
    x: energy offset (MeV)
    sign: narrow smooth or sigexpt, >= sigexpt
    sigw: wide smooth
    dE: level densityspectrum bin width
    varn: relative noise variance parameter
    """
    sigtsq=sigw**2
    A=(dE*varn)/(2.0*np.sqrt(np.pi))
    act=A*( np.exp(-x*x/(4.*sigw*sigw))/sigw-
            np.sqrt(8.)*np.exp(-x*x/(2.*sigtsq))/sigw
            )
    return act
    
def autocorrelation(Gw):
    """
    calculate autocorrelation using Wiener-Kinchin thm.
    return: Ac -- autocovariance
            pwrs -- 'power spectrum', (simple).
    """
    Nfg=len(Gw)
    fg=fft.fft(Gw-np.mean(Gw))
    meand=np.mean(Gw)
    pwrs=(fg*np.conjugate(fg)).real
    Accomplex=(fft.ifft(fg*np.conjugate(fg))) / float(Nfg) #- 1.0
    Ac=Accomplex.real/meand**2
    return Ac, pwrs

def crosscorrelation(Gw,Gv):
    """
    calculate autocorrelation.
    return: Ac -- crosscorrelation
            pwrs -- 'power spectrum', (simple).
    """
    Nfg=len(Gw)
    fg=fft.fft(Gw-np.mean(Gw))
    fh=fft.fft(Gv-np.mean(Gv))
    meand=np.mean(Gw)
    meandv=np.mean(Gv)
    pwrs=(fg*np.conjugate(fh)).real
    Accomplex=(fft.ifft(fh*np.conjugate(fg))) / float(Nfg) #- 1.0
    Ac=Accomplex.real #/(meand*meandv)
    return Ac, pwrs

    

class Spectrum(object):
    """
    Spectrum:        hold basic data for spectrum read in from file.

    Parameters
    ----------

    e:         ndarray  energy of spectrum bins (in MeV)
    g:         ndarray  spectrum data
    filepath:  str  filename (for record)  
    """
    
    def __init__(self, e, spectrum, filepath):

        if len(e) != len(spectrum):
            raise(ValueError, "arrays do not have the same length")
        self.filepath = filepath
        self.energy = e
        self.data = spectrum
        self.deMeV = e[2]-e[1]  # delta E in MeV
        self.dekeV = self.deMeV*1000.0
        self.size = len(self.data)

class RatioSpectrum(Spectrum):
    """
    RatioSpectrum:   generate ration of smoothed spectrum for LD analysis.
   
    Parameters
    ----------

    e:         ndarray  energy of spectrum bins (in MeV)
    g:         ndarray  spectrum data
    filepath:  str  filename (for record) 
    ROIlo:     index of low limit of region of interest of spectrum
    ROIhi:     index of high limit of region of interest of spectrum
    """

    def __init__(self, e, g, filepath, ROIlo, ROIhi):

        super().__init__( e, g, filepath)

        self.ROIlo = ROIlo
        self.ROIhi = ROIhi
        self.Exlo = self.energy[ROIlo]
        self.Exhi = self.energy[ROIhi]
        self.ExROI = self.energy[ROIlo:ROIhi]

    def makeRatioSpectra(self, signarrow, sigwide, energyunit='keV'):
        """
        Make ratio spectra.
        This is separated out as it may be used several times with different parameters.
        Parameters
        ----------
        
        signarrow: float - narrow smoothing standard deviation in keV
        sigwide:   float - wide smoothing standard deviation in keV
        energyunit: str  - units for energy -- default 'keV'
        """

        if energyunit=='keV':
            de = self.dekeV
        elif energyunit == 'MeV':
            de = self.deMeV
        else:
            raise ValueError("Unknown energy units")
        ROIlo = self.ROIlo
        ROIhi = self.ROIhi

        #print(sigwide/de)
        gw = sim.convolute(self.data, sim.gauss, sigwide/de)
        gw[gw<0.2]=0.2  # lower limit (prevents div by zeros) needs more thought ...
        self.gwide = gw
        self.gnarrow = sim.convolute(self.data, sim.gauss, signarrow/de)
        
        self.gROI = self.data[ROIlo:ROIhi]   # raw spectrum
        self.gROIw = self.gwide[ROIlo:ROIhi] # wide
        self.gROIn = self.gnarrow[ROIlo:ROIhi] # narrow

        self.ratioraw = self.gROI / self.gROIw  # raw/wide
        self.rationar = self.gROIn / self.gROIw # narrow/wide
        self.Eratio = self.energy[ROIlo:ROIhi]

    def doAutocorrelation( self ):
        """
        perform autocorrelation (well, autocovariance ..) of full ROI

        Returns
        -------
        
        ACn : ndarray  - AC of narrow smoothed
        ACr : ndarray  - AC of raw ratio 

        Notes
        -----
        Pad with zeros to avoid circular AC
        """
        ACn, pn = autocorrelation( self.rationar )
        ACr, pr = autocorrelation( self.ratioraw )
        return (ACn , ACr, pn.real, pr.real)


class ACBinner(object):
    """
    Generate AC bins for analysis

    Parameters
    ----------

    NAC : int - number of segments in ROI
    Noffset : int - number of offsets per AC segment
    Nslides : int - number of AC window slides per segment
    data : RatioSpectrum  - instance containing data
    """

    def __init__(self, data, NAC=4, Noffset=1, Nslides=5):

        self.data=data
        self.NAC=NAC
        self.Noffset=Noffset
        self.Nslides=Nslides
        self.Ex=self.data.ExROI
        lendata=len(data.gROI)
        dAC=lendata//NAC
        offset=dAC//Noffset
        slide=dAC//Nslides
        aclo,achi=(0,dAC)
        lohi=[]
        # well, do not use Noffset
        while achi<=lendata:
            lohi.append((aclo,achi))
            aclo+=slide
            achi+=slide
        Nbins=len(lohi)
        self.Nbins=Nbins
        self.lohi=lohi
        self.offset=offset
        self.slide=slide
        self.lenAC=dAC
        #print(lohi)

    def get_bins(self):
        return self.lohi

    def get_bin_limits(self, index):
        if index < -len(self.lohi) or index >=len(self.lohi):
            raise ValueError("Index out of range")
        l,h=self.lohi[index]
        return (self.Ex[l],self.Ex[h-1])

    def genBins(self):
        for (l,h) in self.lohi:
            newbin=Bin(self.data, l, h)
            yield newbin

class Bin(object):
    def __init__(self, data, l, h):
        self.low=l
        self.high=h
        self.data=data
        self.dataraw=self.data.ratioraw[l:h]
        self.datanar=self.data.rationar[l:h]
        self.acraw=sim.autocorrelation(self.dataraw)
        self.acnar=sim.autocorrelation(self.datanar)
        self.energy=data.energy[l:h]
        self.Eoffset=self.energy-self.energy[0]

    def get_noise_correction(self, signoise, sigsm):
        de=self.energy[1]-self.energy[0]
        varnoise0 = self.acraw[0]-self.acraw[1]
        #print(de, signoise, sigsm, varnoise0)
        varnoisesm = varnoise0/(2.0*np.sqrt(np.pi)*(signoise/de))
        noisecorrection = varnoisesm*signoise*AC_shape(self.Eoffset, signoise, sigsm)
        return noisecorrection
         
        
                   
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

import numpy.fft as fft
def autocorrelation(Gw,pad=False):
    """
    calculate autocorrelation.
    """
    meand=np.mean(Gw)
    Nfg=len(Gw)
    if pad:
        Gww=np.ones(2*Nfg)*meand
        Gww[0:Nfg]=Gw
    else:
        Gww=Gw
    fg=fft.fft(Gw-np.mean(Gww))
    pwspec=fg*np.conjugate(fg)
    Accomplex=(fft.ifft(pwspec.real)) / float(Nfg) #- 1.0
    Ac=Accomplex.real/meand**2
    return Ac, pwspec

def linautocorrelation(Gw0):
    """
    calculate autocorrelation.
    """
    N=len(Gw0)
    Ac=np.zeros(N)
    Gw=Gw0-np.mean(Gw0)
    for i in range(N):
        ac=np.sum(Gw[i:N]*Gw[0:N-i])/N
        Ac[i]=ac
        
    return Ac

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    nucleus="154Sm"
    xnucleus=nucleus[3:]+nucleus[0:3]  # fiddle name and A !
    path=findstore(xnucleus)

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
        except:
            print("File not found",targetfilename)
            sys.exit(3)

        sig = 20.0/2.35
        sigsm=3.5*sig
        sigsmn=0.39*2.35*sig
        print(sig,sigsm,sigsmn)

        S=RatioSpectrum( X, G, targetfilename, ROIlo, ROIhi )

        S.makeRatioSpectra( sigsmn, sigsm )

        bins=ACBinner(S)
        genbins=bins.genBins()
        plt.figure(2)
        i=1
        for bin in genbins:
            print(bin.low,bin.high)
            plt.subplot(4,4,i)
            plt.plot(bin.Eoffset, bin.acnar)
            plt.xlim(0.0,bin.Eoffset[-1]/4)
            plt.text(0.8,0.8,"%d"%i,transform=plt.gca().transAxes)
            i+=1
        
        ACn, ACr, pn, pr = S.doAutocorrelation()

        e=S.Eratio-S.Eratio[0]
        dE=S.Eratio[1]-S.Eratio[0]
        kfft=fft.fftfreq(len(pn),d=dE)
        # fft returns a 'frequency' -> *2pi needed actual k
        kfft=kfft*2.0*np.pi
        sg=sigsmn/1000.0
        gsian=2.0*np.exp(-kfft*kfft*sg*sg/2.0)
        zl,ze=findzeros(e,ACn)
        emin,emax,lmin,lmax=findminmax(e,ACn)
        
        plt.figure(1)
        plt.plot( e, ACn )
        #plt.plot( S.Eratio-S.Eratio[0], ACr )
        #plt.plot(S.Eratio,S.ratioraw)
        for z in emin:
            plt.axvline(z)
        plt.axhline(0.0,alpha=0.3)
        lmin=np.array(lmin)
        lmax=np.array(lmax)
        avmax=np.mean(lmax)
        avmin=np.mean(lmin)
        plt.axhline(avmax, alpha=0.3,color='r')
        plt.axhline(avmin, alpha=0.3,color='g')
        """
        plt.plot(kfft,pn)
        #plt.plot(kfft,pr)
        plt.plot(kfft,gsian)
        plt.xlim(0,1000)
        """

        plt.figure(3)
        l,h=bins.lohi[0]
        bin=Bin(bins.data,l,h)
        
        plt.plot(bin.Eoffset, bin.acnar)
        plt.xlim(0.0,bin.Eoffset[-1]/4)
        plt.text(0.8,0.8,"%d"%i,transform=plt.gca().transAxes)

        
        plt.show()
