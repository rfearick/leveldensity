import pylab as plt
import numpy as np
import simlib1 as sim
import leveldensities as lden

"""
(19/12/2017)
Copied from uct2017b/Sm154
"""

def PlotSpectrum(nfig,X,G,Gs,ROIlo,ROIhi,A,incounts=True):
    incounts=True
    curfig=plt.figure(nfig)
    curfig.clf()
    # plot spectrum
    #plt.subplot(221)
    plt.plot(X,G,'b-',drawstyle='steps-mid')
    plt.xlabel('Energy [MeV]')
    if incounts:
        plt.ylabel('counts per channel')
    else:
        plt.ylabel(r'$d^2\sigma/d\Omega dE$ [mb/sr MeV]')
    # plot smoothed
    plt.plot(X,Gs,'r-')
    plt.xlim(X[0],X[-1])
    plt.legend(('spectrum','smooth'))
    #plt.axvline(X[ROIlo+AClo])
    #plt.xlim(X[ROIlo],X[ROIhi])
    plt.axvline(X[ROIlo+0], color="m")
    plt.axvline(X[ROIhi+0], color="m")
    ## plt.axvline(X[ROIlo+AChi-1], color="m")
    plt.text(0.1,0.8,r"$^{%i}$Sm"%A,fontsize=16,transform=plt.gca().transAxes)
    plt.title("Data")
    #if M1fractions: plt.plot(em1,m1fraction*10000,'g-',drawstyle='steps-mid')
    #plt.ylim(0.0,20000)


def PlotACs(nfig, i, X,Ac, Act, NACbins, Nac, meanE, targetname, NarrowSmooth):
    plt.figure(nfig)
    plt.suptitle(targetname)
    axx=plt.subplot(int(NACbins/3+0.7),3,1+i)
    Nacd2=Nac#//2
    plt.plot(X[0:Nacd2]-X[0],Ac[0:Nacd2],'b-',drawstyle='steps-mid',label="mean E=%4.1f"%(meanE,))
    if not NarrowSmooth: plt.plot(X[0:2]-X[0],Ac[0:2],'bo')
    plt.xlabel('Energy offset [MeV]')
    plt.ylabel('Autocorrelation')

    # plot Hansen formula
    plt.plot(X[0:Nacd2]-X[0],Act[0:Nacd2],'r-')
    plt.xlim(X[0]-X[0],X[Nacd2-1]-X[0])
    plt.legend()

def PlotFullAC(nfig, Xratio,allAC,popt,alpha,sig,sigsm,initialD,A):
    plt.figure(nfig)
    ax=plt.subplot(211)
    plt.plot(Xratio[0:120]-Xratio[0], allAC[0:120],drawstyle='steps-mid') # was 100
    #plt.plot(Xratio[0:120]-Xratio[0], allAC2[800:920]+0.0000,'c-',drawstyle='steps-mid') # was 100
    plt.plot(Xratio[0:120]-Xratio[0], sim.Hansen2(Xratio[0:120]-Xratio[0], popt[0],popt[1],alpha,sigsm),'g-')
    plt.plot(Xratio[0:120]-Xratio[0], sim.Hansen2(Xratio[0:120]-Xratio[0], initialD,sig,alpha,sigsm),'m-',alpha=0.6)
    plt.text(0.8,0.8,r"$^{%i}$Sm"%A,fontsize=16,transform=ax.transAxes)
    plt.title(r"Autocorrelation of full region of interest")
    plt.ylabel("Autocorrelation")
    plt.xlabel("Energy offset [MeV]")
    plt.subplot(212)
    plt.plot(Xratio[0:750]-Xratio[0], allAC[0:750],drawstyle='steps-mid') # was 100
    plt.plot(Xratio[0:750]-Xratio[0], sim.Hansen2(Xratio[0:750]-Xratio[0], popt[0],popt[1],alpha,sigsm))

    plt.ylabel("Autocorrelation")
    plt.xlabel("Energy offset [MeV]")


def PlotLevelDensities(nfig,N,Z,A,Ed,Dd,Ddl,Ddh,Ex, Exlo, Exhi):
    Spin=1.0
    rhoRauscher=lden.LevelDenRauscher(Ex, Z, N, Spin)
    rhovonEg=lden.LevelDenvonEgidy(Ex, Z, N, Spin)
    rhoGorf=lden.HFBCS(Z,N,Spin,'-')
    rhoGor=rhoGorf(Ex)
    # level density plot
    curfig=plt.figure(nfig)
    curfig.clf()
    yerrs=np.array([Ddl,Ddh])
    #if len(Ed)<1 or len(Dd)<1: continue
    plt.errorbar(Ed,Dd,yerr=yerrs,fmt='o',label='Data')
    ###plt.plot(Ed,Dr,'co',label='alt.') # direct from AC[iAC]
    plt.gca().set_yscale('log')
    ###plt.semilogy(Ed,Dd,'bo')
    plt.semilogy(Ex,rhoRauscher,'g-',label='BSFG (Rauscher)')
    plt.semilogy(Ex,rhovonEg,'g--',label='vonEgidy')
    plt.semilogy(Ex,rhoGor,'m-',label='HFB')
    plt.xlabel('Excitation Energy [MeV]')
    plt.ylabel(r'Level density [Mev$^{-1}$]')
    plt.xlim(Exlo,Exhi)
    ##plt.xlim(11.0,20.0) #40Ca
    plt.text(0.1,0.8,r"$^{%i}$Sm(p,p')"%A,fontsize=18,transform=plt.gca().transAxes)
    plt.text(0.1,0.7,r"Preliminary",fontsize=22,color='r',transform=plt.gca().transAxes)
    #plt.text(0.1,0.7,
    #         r"$\sigma=%5.3f, \sigma_</\sigma=%5.3f, \sigma_>/\sigma=%5.3f$"%(sigexpt,sigsmn/sigexpt/2.3,sigsm/sigexpt),
    #         transform=plt.gca().transAxes)
    plt.legend(loc=4)
    plt.title(r"Level densities")
    #plt.savefig(os.path.join(outputdirname,"%s%sfig3.png"%(targetbase,targetangle)))


    ## # histogram fluctuations
    ## plt.subplot(222)
    ## #plt.hist(Gratio,bins=20)
    ## # plot spectrum
    ## plt.plot(X,G,'b-')
    ## plt.xlabel('Energy [MeV]')
    ## plt.ylabel(r'$d^2\sigma/d\Omega dE$ [mb/sr MeV]')
    ## #plt.ylabel('counts per channel')
    ## # plot smoothed
    ## plt.plot(X,Gs+bkg,'r-')
    ## plt.plot(X,bkg,'g-')
    ## plt.xlim(X[0],X[-1])
    ## plt.ylim(min(G),max(G[ROIlo:ROIlo+AChi-1]))
    ## ###plt.legend(('spectrum','smooth','background'))
    ## #plt.axvline(X[ROIlo+AClo])
    ## plt.axvline(X[ROIlo+0], color="m")
    ## plt.axvline(X[ROIlo+AChi-1], color="m")

    ## plt.savefig(os.path.join(outputdirname,"%s%sfig1.png"%(targetbase,targetangle)))

def PlotAnalysisROI(nfig, Xratio, TG, TGs, TGn, A):
    plt.figure(nfig)
    #plt.plot(Xratio,Gratio)
    plt.plot(Xratio,TG,drawstyle='steps-mid',label='data')
    plt.plot(Xratio,TGs,label='smoothed')
    if TGn is not None:
        plt.plot(Xratio,TGn,label='narrow sm.')
    plt.xlabel("Excitation energy [MeV]")
    plt.ylabel("Cross section")
    plt.ylabel("Counts per 5 keV")
    plt.title("Region of interest")
    plt.text(0.1,0.8,r"$^{%i}$Sm"%A,fontsize=16,transform=plt.gca().transAxes)
    plt.legend(loc=4)
    #plt.plot(Xratio,G0[ROIlo:ROIhi])
    #plt.xlim(8,25)
    #plt.ylim(0,15)

def PlotStationarySpectrum(nfig, Xratio, TG0, TS, A):
    plt.figure(nfig)
    Gratio = TG0 if TS==None else TG0/TS
    plt.plot(Xratio,Gratio)
    plt.xlabel(r"Excitation energy [MeV]")
    plt.ylabel(r"$g_</g_>$")
    plt.text(0.8,0.8,r"$^{%i}$Sm"%A,fontsize=16,transform=plt.gca().transAxes)
    plt.title("stationary spectrum")
    
    """
    plt.figure(19)
    #plot_imfs(imfs)
    plt.subplot(511)
    plt.plot(Xratio,imfs[0])
    plt.subplot(512)
    plt.plot(Xratio,imfs[1])
    plt.subplot(513)
    plt.plot(Xratio,imfs[2])
    plt.subplot(514)
    plt.plot(Xratio,imfs[3])
    plt.subplot(515)
    plt.plot(Xratio,imfs[4])
    """

def PlotTrends(nfig, Ed,Sigfit, ACzero, fitpoints, A):
    plt.figure(nfig)
    plt.subplot(211)
    plt.plot(Ed, Sigfit, 'bo')
    plt.plot(Ed,fitpoints,'g-')
    plt.xlabel(r"Excitation energy [MeV]")
    plt.ylabel(r"Fitted AC sigma")
    plt.subplot(212)
    plt.semilogy(Ed,ACzero,'bo')
    plt.xlabel(r"Excitation energy [MeV]")
    plt.ylabel(r"AC at zero offset")



"""
plt.figure(15)
    yerrs=plt.array([Ddl,Ddh])
    if len(Ed)<1 or len(Dd)<1: continue
    plt.errorbar(Ed,Ddcorr,yerr=yerrs,fmt='o',label='Data')
    ###plt.plot(Ed,Dr,'co',label='alt.') # direct from AC[iAC]
    plt.gca().set_yscale('log')
    ###plt.semilogy(Ed,Dd,'bo')
    plt.semilogy(Ex,rhoRauscher,'g-',label='BSFG (Rauscher)')
    plt.semilogy(Ex,rhovonEg,'g--',label='vonEgidy')
    plt.semilogy(Ex,rhoGor,'m-',label='HFB')
    plt.xlabel('Excitation Energy [MeV]')
    plt.ylabel(r'Level density [Mev$^{-1}$]')
    plt.xlim(Exlo,Exhi)
    ##plt.xlim(11.0,20.0) #40Ca
    plt.text(0.1,0.8,r"$^{%i}$Sm(p,p')"%A,fontsize=18,transform=plt.gca().transAxes)
    plt.text(0.1,0.7,r"Preliminary",fontsize=22,color='r',transform=plt.gca().transAxes)
    #plt.text(0.1,0.7,
    #         r"$\sigma=%5.3f, \sigma_</\sigma=%5.3f, \sigma_>/\sigma=%5.3f$"%(sigexpt,sigsmn/sigexpt/2.3,sigsm/sigexpt),
    #         transform=plt.gca().transAxes)
    plt.legend(loc=4)
    plt.title(r"Level densities")
    #plt.savefig(os.path.join(outputdirname,"%s%sfig3.png"%(targetbase,targetangle)))
"""
