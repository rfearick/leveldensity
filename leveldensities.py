from __future__ import print_function
import numpy as np
import scipy.io as sio


versiondate="23/6/2014"
versionname="leveldensities.py"

"""
(19/12/2017)
Copied from uct2017b/Sm154

Nuclear   Level  Density  Formulas

(23/6/2014)
Taken from: ws2012/rcnp-angle-all-6-dev-uct2.py
               via uct2012b, da2012orig ...

Note:  need to change file op f.next() -> f.readline() for Python 3               
"""


# Level densities ============================

#  First set from SM Grimes, PRC 88, 024613 (2013)

def intrinsic_level_density(U, a):
    """
    Intrinsic state density formula (i.e. basic)
    Grimes (6)
    
    U:  Energy parameter (simple: E_x, usually Ex-backshift)
    a:  level density parameter (simple A/8 MeV**-1)
    """
    rho=(np.sqrt(np.pi)/(12.0*a**0.25))*np.exp(2.0*np.sqrt(a*U))/U**1.25
    return rho
    
def spin_cutoff_level(J, sigma):
    """
    spin cutoff factor for calculating level density, J.
    Grimes (9)
    
    J:      spin
    sigma   (<j^2_z>)**1/2
    """
    cutoff=(1.0/np.sqrt(2.0*np.pi))*(1.0/sigma**3)*2.0*(J+0.5)**2*np.exp(-(J+0.5)**2/(2.0*sigma**2))
    ##print("Basic: sigma=", sigma)
    return cutoff
def spin_cutoff_state(J, sigma):
    """
    spin cutoff factor for calculating state density J,Jz
    Grimes (10)
    
    J:      spin
    sigma   (<j^2_z>)**1/2
    """
    cutoff=(1.0/np.sqrt(2.0*np.pi))*(1.0/sigma**3)*(J+0.5)*np.exp(-(J+0.5)**2/(2.0*sigma**2))
    #cutoff=(1.0/np.sqrt(2.0*np.pi))*(1.0/sigma)*(np.exp(-J**2/(2.0*sigma**2))-np.exp(-(J+1)**2/(2.0*sigma**2)))
    ##print("Basic: sigma=", sigma)
    return cutoff
    
def sigma_calc(E, A, backshift=0.0):
    """
    Taken from below, but corrected (?) 
    """
    a=0.115*A
    U=E-backshift
    ##print("Basic: delta=", backshift)
    ##print("Basic: a=", a)
    return np.sqrt(0.015*A**(5./3.)*np.sqrt(U/a))

def LevelDenBasic(E, backshift, A, spin):
    # factor 0.5 for parity
    return 0.5*intrinsic_level_density(E-backshift,0.115*A)*spin_cutoff_state(spin,sigma_calc(E,A,backshift))


# From Rauscher, Thielemann, Katz, PRC 56, 1613 (1977)

def LevelDenRauscher(E, Z, N, Spin):
    deltan=0.5*(2.0*BEA[Z,N]*(Z+N)- BEA[Z,N-1]*(Z+N-1) - BEA[Z,N+1]*(Z+N+1))
    deltap=0.5*(2.0*BEA[Z,N]*(Z+N)- BEA[Z-1,N]*(Z+N-1) - BEA[Z+1,N]*(Z+N+1))
    backshift=0.5*(deltan+deltap)
    emic = Emic[Z,N]
    return Rauscher(E, Z+N, Spin, backshift, emic)


def Rauscher(E, A, Spin, backshift, Emic):
    # The best fit to experimental data
    # PRC 56 1613
    r0 = 1.2
    alpha = 0.1337     
    betha = -0.06571   
    gamma = 0.04884     
    # Check the back shift
    ##print("Rauscher: delta=", backshift)
    U = E - backshift
    if ( U.all() >= 0.0):
        # Calculate level density parameter
        a = alpha*A+betha*A**(2.0/3.0)
        # Take into account energy dependence
        a = a*(1.0 + Emic*(1.0 - np.exp(-1.0*gamma*U))/U)
        ##print("Rauscher: a=", a)
        # Rigid body momentum
        sigma2 = 0.0096*r0**2*A**(5./3.)*np.sqrt(U/a)
        # Spin cutoff parameter
        #print("Basic: sigma=", np.sqrt(sigma2))
        spincut = 0.5*(2*Spin + 1.0)/sigma2
        spincut = spincut*np.exp(-0.5*Spin*(Spin + 1.0)/sigma2)
        # Level density
        #print U, spcut, sigm2,a
        rho = spincut*np.exp(2.0*np.sqrt(a*U))/U**1.25
        rho = 0.5*rho/(12.0*np.sqrt(2.0*sigma2)*a**0.25)
    else:
        # Not 0.0 because quadl crashs if 2 input values are equal to each
        # other. Even so the result of the integration is completely wrong.
        # That is why the new version calculate rho for U >0 only
        rho = 0.001*U +0.001
    return rho

# From von Egidy & Bucurescu, PRC 72 044311 (2005)
# Corrected in                PRC 73 049901 (2006)

def LevelDenvonEgidy(E, Z, N, Spin):
    Sd=-AM[Z,N]+AM[Z-1,N-1]+AM[1,1]
    Sdplus=-AM[Z+1,N+1]+AM[Z,N]+AM[1,1]
    Pd=0.5*(-1)**(Z+1)*(-Sdplus+Sd)
    M=AM[Z,N]
    Mplus=AM[Z+1,N+1]
    Mminus=AM[Z-1,N-1]
    Pairing=Pd
    mass=Z+N
    return vonEgidy(E, Z, N, Spin, M, Mplus, Mminus, Pairing)
                    
def vonEgidy(E, Z, N, Spin, M, Mplus, Mminus, Pairing):
    # PRC 72, 044311; PRC 80, 054310
    Mn = 939.565
    MH = 938.783
    ee = 1.602E-19
    k = 8.987E+9
    jMeV = 1.602E-13
    A = Z+N
    # Liquid drop formula
    r0 = 1.233E-15
    avol = -15.65
    asf = 17.63
    asym = 27.72
    ass = -25.60
    # Level density parameter
    pa1 = 0.127
    pa2 = 4.98E-3
    pa3 = -8.95E-5
    # Energy backshift
    # Original PRC publication
    #pe1 = -0.468
    #pe2 = -0.565
    #pe3 = -0.231
    #pe4 = 0.438
    # Erratum
    pe1 = -0.48
    pe2 = -0.57
    pe3 = -0.24
    pe4 = 0.29
    pe5 = 0.70

    # Binding energies, masses and shell corrections
    Eb = avol + asf*A**(-1.0/3.0) + k/jMeV*3.0*ee**2/(5.0*r0)*Z**2*A**(-4.0/3.0) + (asym+ass*A**(-1.0/3.0))*((N-Z)/A)**2
    Eb = Eb*(N + Z)
    Mld = N*Mn + Z*MH + Eb
    S = M - Mld
    Eb = avol + asf*(A+2)**(-1.0/3.0) + k/jMeV*3.0*ee**2/(5.0*r0)*(Z+1)**2*(A+2)**(-4.0/3.0) + (asym+ass*(A+2)**(-1.0/3.0))*((N-Z)/(A+2))**2
    Eb = Eb*(A+2)
    Mld = (N+1)*Mn + (Z+1)*MH + Eb
    Splus = Mplus - Mld
    Eb = avol + asf*(A-2)**(-1.0/3.0) + k/jMeV*3.0*ee**2/(5.0*r0)*(Z-1)**2*(A-2)**(-4.0/3.0) + (asym+ass*(A-2)**(-1.0/3.0))*((N-Z)/(A-2))**2
    Eb = Eb*(A-2)
    Mld = (N-1)*Mn + (Z-1)*MH + Eb
    Sminus = Mminus - Mld
    dSdA = 0.25*(Splus - Sminus)
    # Type of the nucleous
    if (not(Z&1)):
        evenZ = 1
    else :
        evenZ = 0
    if (not(N&1)):
        evenN = 1
    else:
        evenN = 0
    # Original publication PRC
    # Pairing correction and energy backshift
    #if (evenZ == 1 && evenN == 1)
    #    delta = 0.5*Pairing
    #    E1 = pe1 - 0.5*Pairing + pe4*dSdA 
    #elseif (evenZ == 0 && evenN == 0)
    #    delta = -0.5*Pairing
    #    E1 = pe3 + 0.5*Pairing + pe4*dSdA 
    #else
    #    delta = 0
    #    E1 = pe2 - 0.5*Pairing + pe4*dSdA 
    #end
    # Erratum
    # Pairing correction and energy backshift
    if (evenZ == 1 and evenN == 1):
        delta = 0.5*Pairing
        backshift = pe1 - 0.5*Pairing + pe4*dSdA 
    elif (evenZ == 1 and evenN == 0):
        delta = 0.0
        backshift = pe2 - 0.5*Pairing + pe5*dSdA 
    elif (evenZ == 0 and evenN == 1):   
        delta = 0.0
        backshift = pe2 + 0.5*Pairing - pe5*dSdA 
    else:
        delta = -0.5*Pairing
        backshift = pe3 + 0.5*Pairing + pe4*dSdA # rwf -> was pe3 see erratum 
    # Check the backshift
    U=E-backshift
    if (U.all() >= 0.0):
        # Final shell correction
        Sprime = S-delta
        # Level density parameter
        a = (pa1 + pa2*Sprime + pa3*A)*A
        # Spin cutoff parameter
        sigma2 = 0.0146*A**(5.0/3.0)/(2.0*a)
        sigma2 = sigma2*(1.0 + np.sqrt(1.0 + 4.0*a*U))
        spincut = np.exp(-0.5*(Spin**2.0)/sigma2) - np.exp(-0.5*((Spin + 1.0)**2)/sigma2)
##        print("Egidy: a=", a)
##        print("Egidy: delta=", backshift)
##        print("Egidy: sigma=", np.sqrt(sigma2))
        # Level density
        rho = 0.5*spincut*np.exp(2.0*np.sqrt(a*U))/U**1.25
        #print U, fJ, sigm2a, alpha
        rho = rho /(12.0*np.sqrt(2.0*sigma2)*a**0.25)
    else:
        # Not 0.0 because quadl crashs if 2 input values are equal to each
        # other. Even so the result of the integration is completely wrong
        rho = 0.001*U +0.001
    return rho


# From Hillaire & Goriely, Nucl Phys A 779, 63 (2006)

def HFBCS(Z, N, Spin, P):
    """
    Nuclear level density from microscopic calculation with HFB + BSk14 Skyrme force.
    Hillaire & Goriely: NP A 779, 63, (2006)
    Data tables from:
    http://www.astro.ulb.ac.be/pmwiki/Brusslib/Level

    Input:
    Z, N:   Usual proton, neutron numbers
    P:      Parity, "+" or "-" (string)
    """
    import re
    def lister(data):
        for l in data:
            yield l
    
    A=Z+N
    # data is read from a table and interpolated.
    datatable="../../uct2011b/z"+"%03d.tab"%(Z,)
    print(datatable)
    f=open(datatable,"r")
    # now must extract isotope with right parity
    # use regular expressions to pattern match headers
    if P=='+':
        r=re.compile("Z= %2d A=%d: P"%(Z,A))
    else:
        r=re.compile("Z= %2d A=%d: N"%(Z,A))
    #r=re.compile("Z")
    for l in f:
        #print(l)
        m=r.search(l)
        if m == None: continue
        #print(l)
        #print(m.group())
        l=f.readline()
        #print("a",l)
        l=f.readline()
        l=f.readline()
        lines=[]
        #print(l)
        while (len(l)>2):
            lines.append(l)
            l=f.readline()
        break
    f.close()
    # dump data to file so we can unpack into array (easier?)
    g=lister(lines)
    data=np.loadtxt(g,unpack=True)
    # data: [0] Energy U (MeV), [6] rho J=1, [7] rho J=2, etc.
    indexspin=np.rint(Spin)+5
    indexspin=int(indexspin)
    # interpolate for plotting
    from scipy.interpolate import interp1d
    if  P=='+':
        return interp1d(data[0],data[indexspin])
    else: 
        return interp1d(data[0],data[indexspin])

##def lister(data):
##    for l in data:
##        yield l
##    f=open("tmpl.xxx","w")
##    f.writelines(data)
##    f.close
##    return "tmpl.xxx"
    
# end level densities ==============================


    # now plot level densities,  expt and theory as fn. of E

# initialize level density data
mfile=sio.loadmat("../../da2014/MassTable.mat")
BDE=mfile["BDE"]
NM=mfile["NM"]
Emic=mfile["Emic"]
BEA=mfile["BEA"]
AM=mfile["AM"]
Mexcess=mfile["Mexcess"]


if __name__ == "__main__":

    import pylab as plt

    def Open( name, ndec=0 ):
        """
        open spectrum file and optionally decimate
        returns 2048 channel spectrum
        """
        en,a,da=np.loadtxt("../../uct2011b/data/5keV/"+name, unpack=True)
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



    Ex=np.linspace(10.0,25.0, 200)
    # init level density theory
    Z=20
    N=20

    Spin=1.0
    #rhoRauscher=Rauscher(Ex, mass, Spin, delta, emic)
    #rhovE=vonEgidy(Ex, Z, N, Spin, M, Mplus, Mminus, Pairing)
    rhoRauscher=LevelDenRauscher(Ex, Z, N, Spin)
    rhovE=LevelDenvonEgidy(Ex, Z, N, Spin)
    rhoGrimes=LevelDenBasic(Ex, 3.0, Z+N, Spin)
    try:
        rhoGf1=HFBCS(Z,N,Spin,'-')
        rhoGf2=HFBCS(Z,N,Spin+1,'+')
        rhoG1=rhoGf1(Ex)
        rhoG2=rhoGf2(Ex)
        #print "HF",targetbase,rhoG1
        doHF=True
    except:
        print("No HFB")
        doHF=False
    #print rhoRauscher
    #print rhovE
    #print Ex

    plt.rc('text',usetex=True)
    plt.rc('mathtext',default='rm')
    plt.rc('font', family='serif', size=12.0)
    plt.rc('figure.subplot',top=0.94, right=0.95)
    plt.rc('legend',fontsize='small')
    plt.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})


    # level density plot
    curfig=plt.figure()
    curfig.clf()
    #plt.gca().set_yscale('log')
    ##    plt.semilogy(Ed,Dd,'bo')
    plt.semilogy(Ex,rhoRauscher,'g-',label='BSFG (Rauscher)')
    plt.semilogy(Ex,rhovE,'b-',label='BSFG (vonEgidy)')
    plt.semilogy(Ex,rhoGrimes,'r-',label='BSFG (Grimes)')
    if doHF:
        plt.semilogy(Ex,rhoG1,'m-',label='HFB 1-')
        plt.semilogy(Ex,rhoG2,'m--',label='HFB 2+')
    plt.xlabel('Energy [MeV]')
    plt.ylabel(r'Level density [Mev$^{-1}$]')
    plt.xlim(9.0,26.0)
    plt.legend(loc=4)

    plt.show()
