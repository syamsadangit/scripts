#!/usr/bin/env python 
# -*- coding: utf-8 -*- 

from ase.calculators.vasp import Vasp
import mmap
import inspect as ins
from ase.io import read, write
from ase.io.vasp import read_vasp_out as rdout
import pandas as pd
import os, re
import shutil
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date
from scipy.optimize import curve_fit

##################################################
#DEFINE FUNCTIONS HERE
##################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def extract_and_save_columns(inF, outF, colReq):
    df = pd.read_csv(inF, delim_whitespace=True)
    df_selected = df[colReq]
    current_directory = os.getcwd()
    df_selected.insert(0, 'Current_Directory', current_directory)
    df_selected.to_csv(outF, sep=' ', index=False, header=True)
    print(f"Data has been written to {outF}")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Logger:
    """Handles logging output to a file."""
    def __init__(self, filename):
        self.filename = filename
        with open(self.filename, 'w') as f:
            f.write("Voltage Analysis Log\n")
            f.write("====================\n\n")

    def write(self, *messages):
        with open(self.filename, 'a') as f:
            for message in messages:
                # Convert message to string if it's not already a string
                f.write(str(message) + "\n")

    def log_fit_results(self, fitFn, popt, pcov):
        with open(self.filename, 'a') as f:
            f.write(f"Fitted Function: {fitFn.__name__}\n\n")
            f.write(
                "Optimal Parameters:\n" + \
                "\n".join(f"  Parameter {i+1}: {p:.6f}" for i, p in enumerate(popt)) + "\n\n"
            )
            f.write(
                "Covariance Matrix:\n" + \
                "\n".join(
                    "  " + " ".join(f"{v:.6e}" for v in row) for row in pcov
                ) + "\n\n"
            )


def fit_and_generate_curve(fitFn, xFitDat, yFitDat, xmin, logger, step=0.03, tolerance=0.001, max_points=1000):
    """
    Fit a function to data, compute fitted values, log results, and generate a curve.

    Parameters:
        fitFn (callable): Function to fit.
        xFitDat (array-like): Independent variable values (input data).
        yFitDat (array-like): Dependent variable values (input data).
        xmin (float): Starting value for the independent variable.
        logger (Logger): Logger instance for saving fit results.
        step (float): Increment for fitted x-values.
        tolerance (float): Range to stop generating y-values.
        max_points (int): Max points for the fitted curve.

    Returns:
        (np.array, np.array, np.array, np.array): Fitted x, fitted y, parameters, and covariance.
    """
    # Fit the function to the provided data
    popt, pcov = curve_fit(fitFn, xFitDat, yFitDat)
    
    # Log fit results
    logger.log_fit_results(fitFn, popt, pcov)
    
    # Calculate the corresponding minimum y-value at xmin
    ymin = fitFn(xmin, *popt)
    
    cnt = 0
    x0 = xmin
    if isinstance(yFitDat, np.ndarray):
        y_xmax = yFitDat[-1]
    elif isinstance(yFitDat, pd.Series):
        y_xmax = yFitDat.iloc[-1]
    ycut = np.min([ymin,y_xmax])
    x = []
    y = []

    # Generate fitted curve based on the parameters
    while cnt == 0:
        y0 = fitFn(x0, *popt)
        y.append(y0)
        x.append(x0)
        
        # Control num of data points
        if x0 != xmin:
            if y0 < ycut: 
                cnt = 1
            elif len(y) >= max_points:
                cnt = 1
        
        # Increment x0 by the step size
        x0 = x0 + step

    # Convert lists to numpy arrays
    x = np.array(x)
    y = np.array(y)

    return x, y, popt, pcov

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def fitFn(x, a0, a1, a2):
    y = a0 + a1*x + a2*(x**2) # + a3*(x**3)          
    return y        

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_vol(x, yL1, mkr1='o', lsty1='-' ,\
            xL2=None, yL2=None, mkr2='o',lsty2='--' ,\
            xL3=None, yL3=None, mkr3='o',lsty3='-',\
            xL4=None, yL4=None, mkr4='o',lsty4='-',\
            xlbl='x-axis', ylbl='y-axis', titl='Plot of x vs y',\
            figname=None,secAxis=False,secXL=None):
    #yLx are lists with [yx, legendx]
    fig = plt.figure(figsize=(12,7))
    ax1 = fig.add_subplot(111) 
    if not figname:
        time = datetime.now().strftime("%H-%M-%S")
        date = datetime.now().date().isoformat()
        figname = date+'-'+time
    ax1.plot(x,yL1[0], label=yL1[1], marker=mkr1,linestyle=lsty1)
    if yL2:
        ax1.plot(xL2,yL2[0], label=yL2[1], marker=mkr2,linestyle=lsty2)
    if yL3:
        ax1.plot(xL3,yL3[0], label=yL3[1], marker=mkr3,linestyle=lsty3) 
    if yL4:
        ax1.plot(xL4,yL4[0], label=yL4[1], marker=mkr4,linestyle=lsty4) 
    ax1.set_xlabel(xlbl)
    ax1.set_ylabel(ylbl)
    ax1.set_title(titl)
    ax1.legend()
    if secAxis:
        if len(secXL[0])==len(x):
            ax2 = ax1.twiny()
            new_tick_locations = x 
            ax2.set_xlim(ax1.get_xlim())
            ax2.set_xticks(new_tick_locations)
            ax2.set_xticklabels(np.round(secXL[0],2),rotation=90)
            ax2.set_xlabel(secXL[1])
        else:
            logger.write("The length of secondary x and primary x values are different.\n")
    plt.savefig(figname)
    plt.close()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def is_float(element):
    # If you expect None to be passed:
    if element is None: 
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def dir_filter():
    pattern = r"dnel.*\."
    path = os.getcwd()
    lstAll = os.listdir(path)
    dirLst = [d for d in lstAll if os.path.isdir(os.path.join(path, d))]
    dirFilter = [d for d in dirLst if re.search(pattern, d)] 
    nelstr = [dele.strip('dnel') for dele in dirFilter]
    nelLst = []
    for ele in nelstr:
        if is_float(ele):
            nelLst.append(float(ele))
    nelLst.sort()
    return nelLst
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def make_incar(INCAR, crctn):
    #THE FUNCTION INPUTS INCAR AND CORRECTIONS
    #IT OUTPUTS NEW INCAR WITH THE OLD INCAR
    #AND CORRECTIONS WITH THE OLD PARAMETERS
    #CHANGED TO NEW PARAMETERS 
    INCARcp = INCAR.copy()
    for parameter, value in crctn.items():
        INCARcp[parameter] = value
    return INCARcp
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def SPINMAGMOM(atoms, spin_crctn, INCAR, dir):
    #SETS MAGMOM OF ATOMS 
    for atom in atoms:
        if atom.symbol == 'Ni':
            atom.magmom = 2.0
        elif atom.symbol == 'P':
            atom.magmom = 1.0
        elif atom.symbol == 'H':
            atom.magmom = 1.0
        elif atom.symbol == 'O':
            atom.magmom = 1.0
    INCARSPIN = make_incar(INCAR, spin_crctn)
    Vasp.xc_defaults['incspin'] = INCARSPIN
    vaspcalc = Vasp(xc='incspin', directory=dir)
    atoms.calc = vaspcalc
    logger.write('~~~~~~~~~~~~~~~~~~~~~~~~~~')
    logger.write('Starting spin calculations')
    logger.write('--------------------------')
    EDFTspin = atoms.get_potential_energy()
    logger.write('Spin calculations ended')
    logger.write('~~~~~~~~~~~~~~~~~~~~~~~\n')
    return EDFTspin, INCARSPIN 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def read_nelect(OUTCAR):
    #READ NELECT AND ENERGY FROM OUTCAR 
    readOUT = rdout(OUTCAR)
    energy0 = readOUT.get_potential_energy()
    with open(OUTCAR,'r') as out:
        for line in out:
            if('NELECT' in line):
                tmp_lst = line.strip().split()
    for str in tmp_lst:
        try:
            num = float(str)
        except ValueError:
            continue
        else:
            nelect0 = num 
            logger.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            logger.write(" The NELECT before voltage calculations is {}".format(num))
            logger.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
    return nelect0, energy0
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def cp_out(i,atoms,dir):
    atoms.write(dir+'/'+'POSCAR{}'.format(i),sort=False, direct=True, vasp5=True) 
    shutil.copy(dir+'/'+'OUTCAR', dir+'/'+'OUTCAR{}'.format(i))
    shutil.copy(dir+'/'+'OSZICAR', dir+'/'+'OSZICAR{}'.format(i))
    shutil.copy(dir+'/'+'vasp.out', dir+'/'+'vasp{}.out'.format(i))
    shutil.copy(dir+'/'+'CONTCAR', dir+'/'+'CONTCAR{}'.format(i))
    shutil.copy(dir+'/'+'INCAR', dir+'/'+'INCAR{}'.format(i))
    shutil.copy(dir+'/'+'KPOINTS', dir+'/'+'KPOINTS{}'.format(i))
    logger.write(f'{dir} cp done\n')
    return
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def read_efermi_fermishift(OUTCAR,vaspout):
    with open(OUTCAR,'r') as out:
        search = b'Fermi energy'
        m = mmap.mmap(out.fileno(), 0, prot=mmap.PROT_READ)
        i = m.rfind(search)
        m.seek(i)
        line = str(m.readline(), encoding="utf-8").strip().split()
    for wrd in line:
            try:
                num = float(wrd)
            except ValueError:
                continue
            else:
                efermi = num
                logger.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                logger.write(" The Fermi energy is {}".format(efermi))
                logger.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
    out.close()
    with open(vaspout,'r') as vspout:
        search = b'FERMI_SHIFT'
        m = mmap.mmap(vspout.fileno(), 0, prot=mmap.PROT_READ)
        i = m.rfind(search)
        m.seek(i)
        line = str(m.readline(), encoding="utf-8").strip().split()
    for wrd in line:
        try:
            num = float(wrd)
        except ValueError:
            continue
        else:
            fermishift = num
            logger.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            logger.write(" The Fermi shift is {}".format(fermishift))
            logger.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
    vspout.close()
    return efermi, fermishift
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def run_vasp(loop_lst, atoms, incar_dct, delta_nelect=None, voltage_flag=False, spin_flag=False, vol_spin_flag=False, spin_crctn=None):
    #voltage_flag is boolean and enables voltage calculations
    #spin_flag is boolean and enables spin polarised calculations
    #delta_nelect is a list of change in num of electrons 
    #if(sol_flag or sol_vol_flag):
    #    for i in loop_lst:
    #        excorr='in{}'.format(i)
    #        INCkey='INC{}'.format(i)
    #        incar = incar_dct[INCkey]
    #        Vasp.xc_defaults[excorr] = incar 
    #        vaspcalc = Vasp(xc=excorr)
    #        atoms.calc = vaspcalc
    #        EDFT = atoms.get_potential_energy()
    #        cp_out(i,atoms,'.')       
    #else:
    #    incar = INCAR4
    #if(spin_flag): #need to modify this block to make it independent of sol block above
    #    spin_name = 'spin' 
    #    edftspin, incarspin = SPINMAGMOM(atoms, spin_crctn, incar, '.')  
    #    cp_out(spin_name,atoms,'.')
    #if(voltage_flag or sol_vol_flag):
    #    if(sol_flag or sol_vol_flag):
    #        incar = incar
    #    else:
    #        incar = INCAR4            
    OUTCAR = 'OUTCAR' #to calc the PZC by reading energy (dnel = 0)
    vaspout = 'vasp.out' #to find fermi shift at PZC (dnel = 0)
    voltage(delta_nelect, OUTCAR, vaspout, vol_spin_flag, spin_crctn)     
    return
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def voltage( delta_nelect, OUTCAR, vaspout, vol_spin_flag, spin_crctn=None):
    #atoms is Atom object, voltage_flag is boolean
    #delta_nelect is list, INCAR is dict, OUTCAR is filename 
    nelect0, energy0 = read_nelect(OUTCAR)                
    efermi0, fermishift0 = read_efermi_fermishift(OUTCAR, vaspout)
    Efermi, FermiShift, Nelect  = [efermi0], [fermishift0], [nelect0]
    EDFTNele = [energy0]
    del_nele0 = np.array([0])
    DNele = np.concatenate((del_nele0, delta_nelect), axis=0)
    voloutput ='VOLTAGE-OUTPUT.dat' 
    pzcout = 'pzc-nodiople-voltage.dat'
    for dnel in delta_nelect:
        dir = 'dnel{}/results'.format(dnel)
        nelect = nelect0 + dnel
        Nelect.append(nelect)
        vol_crctn = {'nelect':nelect,
                        'ldipol':False}
        vaspcalc = Vasp(restart=True,directory=dir)
        atoms = vaspcalc.get_atoms()
        logger.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        logger.write('Reading voltage calculations with NELECT {}'.format(nelect))
        logger.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
        EDFTnelect = atoms.get_potential_energy()
        EDFTNele.append(EDFTnelect)
        dir_read_OUT = dir+'/'+'OUTCAR'
        dir_read_vsp = dir+'/'+'vasp.out'
        efermi, fermishift = read_efermi_fermishift(dir_read_OUT, dir_read_vsp)        
        Efermi.append(efermi) 
        FermiShift.append(fermishift)
        name_dnel = 'dnel{}'.format(dnel)
        cp_out(name_dnel,atoms,dir)
    EDFTNele = np.array(EDFTNele)
    Efermi, FermiShift, Nelect = np.array(Efermi), np.array(FermiShift), np.array(Nelect)
    DEfermi = Efermi - efermi0
    DFermiShift = FermiShift - fermishift0
    MUe = Efermi + FermiShift #IF FS is -ve +FS, if FS is +ve -FS. Get the sign from x-y avg local potential plot. 
    Uvac = - MUe # in volts
    UvSHE = Uvac - Phi_SHE
    UvSHEvPZC = UvSHE - UvSHE[0] 
    UvPZC = Uvac - UvSHE[0] 
    Urlvnt, Urtag    = Uvac, '#WrkFnc' #UvSHE, '#UvSHE'
    Omega = EDFTNele + (DNele*Urlvnt) + (DNele*FermiShift) 
    vol_dct = {'#Nelect':Nelect, '#DNele':DNele, '#Efermi':Efermi, '#DEfermi': DEfermi,\
                '#FermiShift': FermiShift, '#DFermiShift': DFermiShift, '#MUe':MUe,\
                '#WrkFnc':Uvac,'#UvSHE':UvSHE, '#UvSHEvPZC':UvSHEvPZC, '#UvPZC':UvPZC,\
                '#EDFTNele':EDFTNele, '#Omega': Omega}
    vol_df = pd.DataFrame(vol_dct)    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot relevant data below 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    vol_df_sorted = vol_df.copy()
    vol_df_sort2 = vol_df.copy()
    vol_df_sort2 = vol_df_sort2.sort_values(by=['#DNele'])  
    vol_df.to_csv(voloutput,sep=' ',index=False)
    pzcvol_df = vol_df[vol_df['#DNele']==0.0]
    pzcvol_df.to_csv(pzcout, sep=' ', index=False, header=True)
    plot_vol(vol_df_sort2['#DNele'], [vol_df_sort2['#UvSHE'],'U vs SHE'],\
            xlbl='delta nelect', ylbl='Applied potential (V)', titl='Mapping dnelect to applied electrode potential', figname='dn-U.png')
    plot_vol(vol_df_sort2['#DNele'], [vol_df_sort2['#WrkFnc'],'Work function'],\
            xlbl='delta nelect', ylbl='Work function (eV)', titl='Work function as a function of dnelect', figname='dn-wrkfnc.png')
    plot_vol(vol_df_sort2['#UvSHE'], [vol_df_sort2['#Omega'],'Grand canonical energy'],\
            xlbl='U vs SHE (V)', ylbl='Omega (eV)', titl='Grand canonical energy', figname='dn-omega.png',\
            secAxis=True,secXL=[vol_df_sort2['#DNele'],'delta nelect'])
    vol_df_sorted = vol_df_sorted.sort_values(by=[Urtag])
    wrkfnPZC = vol_df_sorted.loc[vol_df_sorted['#DNele']==0.0,['#WrkFnc']].values 
    omegaPZC = vol_df_sorted.loc[vol_df_sorted['#DNele']==0.0,['#Omega']].values
    xwrkPZC, ywrkPZC = wrkfnPZC[0,0], omegaPZC[0,0]
    logger.write(f'Used value of Phi_SHE (reference) is {Phi_SHE}\n')
    logger.write(f'work function at PZC is {xwrkPZC}')
    logger.write(f'Grand canonical energy at PZC is {ywrkPZC}\n' )
    if vol_df_sorted.shape[0] > 1: 
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        def quadFit(x,C):
            #x is wrkFnc or Uvac
            y = ywrkPZC - 0.5*C*(x - xwrkPZC)**2
            return y  
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #_________________________________________________
        # Plot Expected Omega-U relation 
        #_________________________________________________
        xquad = np.linspace(0,2*xwrkPZC,100)
        for C in np.linspace(0,1.2,10):
            plt.plot(xquad-Phi_SHE,quadFit(xquad,C) , label=f"{C}")
        plt.plot(xwrkPZC-Phi_SHE,ywrkPZC,'ok',label=f'PZC={np.round(xwrkPZC-Phi_SHE,2)} E={np.round(ywrkPZC,2)}')
        plt.xlabel('U v SHE (V)')
        plt.ylabel('Omega (eV)')
        plt.legend()
        plt.savefig('expectedUvOmega.png')
        plt.close()
        #_________________________________________________
        xFitDat, yFitDat = vol_df_sorted[Urtag],vol_df_sorted['#Omega']
        xmin = vol_df_sorted.min(axis=0)[Urtag]

        F1 = ins.getsource(fitFn)
        logger.write(F1,'\n')
        x, y, popt, pcov = fit_and_generate_curve(fitFn, xFitDat, yFitDat, xmin,
                                                  logger, step=0.03, tolerance=0.001,
                                                  max_points=1000)
        fitFnCap = np.round(-2*popt[2],2) 
        logger.write(f'Capacitance = {fitFnCap}\n')
    
        
        F2 = ins.getsource(quadFit)
        logger.write(F2, '\n')
        x4C, y4C, poptC, pcovC = fit_and_generate_curve(quadFit, xFitDat, yFitDat, xmin,
                                                  logger, step=0.03, tolerance=0.001,
                                                  max_points=1000)
        quadFitCap = np.round(poptC[0],2)
        logger.write(f'Capacitance = {quadFitCap}\n')

        #_________________________________________________
        # Plot Omega and Fit function vs U 
        #_________________________________________________
        plot_vol(
        x = x - Phi_SHE,
        yL1 = [y, f"""Quadratic fit of grand canonical energy\nFitted Capacitance = {fitFnCap}"""],
        mkr1 = "None",
        lsty1 = '--',
        xL2 = vol_df_sorted['#WrkFnc'] - Phi_SHE,
        yL2 = [vol_df_sorted['#Omega'], 'Grand canonical energy'],
        lsty2 = "None",
        xL3 = xwrkPZC - Phi_SHE,
        yL3 = [ywrkPZC, 'PZC Free energy'],
        xL4 = x4C - Phi_SHE,
        yL4 = [y4C, f"""Free energy, Potential, Capacitance fit\nFitted Capacitance = {quadFitCap}"""],
        mkr4 = "None",
        lsty4 = '--',
        xlbl = 'U vs SHE (V)',
        ylbl = 'Energy (eV)',
        titl = 'Grand canonical energy as a function of applied potential',
        figname = 'wrkfnc-Omega.png')
    #_________________________________________________
    # Plot Omega, EDFT vs U 
    #_________________________________________________
    plot_vol(
    x = vol_df_sorted['#WrkFnc'] - Phi_SHE,
    yL1 = [vol_df_sorted['#EDFTNele'], 'Electronic free energy (eV)'],
    xL2 = vol_df_sorted['#WrkFnc'] - Phi_SHE,
    yL2 = [vol_df_sorted['#Omega'], 'Grand canonical energy'],
    xlbl = 'U vs SHE (V)',
    ylbl = 'Energy (eV)',
    titl = 'Free energy as a function of applied potential',
    figname = 'U-Fene.png',
    secAxis = True,
    secXL = [vol_df_sorted['#DNele'], 'delta nelect'])
    #_________________________________________________
    # Plot Fermi energy vs U 
    #_________________________________________________
    plot_vol(
    x = vol_df_sorted['#UvSHE'],
    yL1 = [vol_df_sorted['#Efermi'], 'Fermi energy'],
    xlbl = 'U vs SHE (V)',
    ylbl = 'Energy (eV)',
    titl = 'Fermi energy as a function of applied potential',
    figname = 'USHE-Efermi.png')
    #_________________________________________________
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plotting over 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    logger.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    logger.write('Voltage output written to '+voloutput)
    logger.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
    logger.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    logger.write("Voltage calculations finished")
    logger.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
    return
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##################################################



INCAR1 = {'gga' : 'PE',
            'pp' : 'PBE',
            'prec' : 'Accurate',
            'lreal' : 'Auto',
            'encut' : 500,
            'kpts' : (3,3,1),
            'ismear' : 0,
            'sigma' : 0.1,
            'potim' : 0,
            'nelm' : 300,
            'lwave' : True,
            'lcharg' : True,
            'isym' : 0,
            'npar' : 8,
            'algo' : 'Fast',
            'istart' : 1,
            'icharg' : 1,
            'isif' : 2,
            'ibrion' : 3,
            'ediffg' : -0.05,
            'ediff' : 1e-6,
            'nsw' : 100,
            'iopt' : 7,
            'maxmove' : 0.3}

crctn1 = {'lsol' : True,
            'eb_k' : 78.4,
            'lambda_d_k' : 3.0,
            'maxmove' : 0.2,
            'istart':1,
            'icharg':1 }

crctn2 = {'ediffg' : -0.02,
            'ediff' : 1e-7,
            'istart':1,
            'icharg':1 }

crctn3 = {'idipol' : 3, 
            'ldipol' : False,
            'istart':1,
            'icharg':1 }

crctn4 = {'ispin' : 2,
            'maxmove' : 0.15,
            'kpts' : (6,6,1),
            'sigma': 0.05,
            'istart':1,
            'icharg':1 }

#INCAR1 #without VASPSol 
INCAR2  = make_incar(INCAR1, crctn1)    #with VASPSol
INCAR3  = make_incar(INCAR2, crctn2)    #INCAR2 + ediffg 0.02
INCAR4  = make_incar(INCAR3, crctn3)    #INCAR3 + ldipol
#INCAR5 = make_incar(INCAR4, crctn4)    #INCAR4 + spin

INCARS = {'INC1':INCAR1, 'INC2': INCAR2, 'INC3':INCAR3, 'INC4':INCAR4} # add 'INC5':INCAR5 for spin


##################################################
#ASSIGN PARAMETERS BELOW
##################################################

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#ASSIGN SOLVENT CALC PARAMETERS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sol_flag        =   False
loop_lst        =   [1, 2, 3, 4]    #choose incars from INCARS dict
spin_flag       =   False           #enables spin calc
spin_crctn      =   crctn4          #parm dict. for spin incar. 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#VOLTAGE CALC 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
voltage_flag    =   True                #enables voltage calc
delta_nelect    =   dir_filter()        #finds the directories automatically 
vol_spin_flag   =   True                #enables voltage & spin calc
Phi_SHE         =   4.43                #eV (Hannes Pt paper)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#ASSIGN SOL AND VOL CALC PARAMETERS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sol_vol_flag    =   False           #sol(check crctn1 dct) and vol calc
                                    #sol_vol_flag is to do both solvent and voltage calculation from the INCAR1                                    

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#FUNC INPUT PARAMETERS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
system = 'NotApplicable'
logger = Logger("voltage-analysis.dat") #Redirecct stdout
pzclog = Logger("PZC-nodipole.dat")
##################################################
#CALL FUNCTIONS BELOW
##################################################

run_vasp(loop_lst, system, INCARS, delta_nelect, voltage_flag, spin_flag, vol_spin_flag, spin_crctn)


