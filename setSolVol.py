#!/usr/bin/env python 
# -*- coding: utf-8 -*- 

from ase.calculators.vasp import Vasp
import mmap
import fileinput
import fnmatch
from ase.io import read, write
from ase.io.vasp_parsers.incar_writer import write_incar
from ase.io.vasp import read_vasp_out as rdout
from ase.io.vasp import write_vasp 
import pandas as pd
import os
import shutil
import numpy as np
import sys

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#DEFINE FUNCTIONS HERE
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def usage():
    print(" ")
    print("For geometry optimisations with varying N electrons (voltage scan) use options -l or -i")
    print("For Dimer calculations with varying N electrons (voltage scan) use options -dl or -di")
    print(" ")
    print("Usage: setSolVol.py -l/-dl start stop step")
    print("    option -l/-dl for limits. Start, end and step of dnelect dir to be made.")
    print("    eg. setSolVol.py -l/-dl -1.5 1.6 0.3\n")
    print("Usage: setSolVol.py -i/-di dnelect-values")
    print("    option -i/-di for include individual dnelect-values")
    print("    eg. setSolVol.py -i/-di 1.0 -1.0 -1.2\n")
    print("The calculations will request 3 (4) days for geometry optimisation (Dimer calculation). If need to change, change in setSolVol.py \n")
    sys.exit(1)
    return
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def jobScript(input_filename, output_dir, find_replace_dct):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base_filename = os.path.basename(input_filename)
    output_filename = os.path.join(output_dir, base_filename)
    
    with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
        for line in infile:
            original_line = line
            for pattern in find_replace_dct:
                if fnmatch.fnmatch(line.strip(), pattern):
                    line = find_replace_dct[pattern] + '\n'
                    break
            outfile.write(line)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def write_kpoints_file(kpoints, filename='KPOINTS'):
    # Ensure kpoints is a list of three integers
    if not (isinstance(kpoints, (list, tuple)) and len(kpoints) == 3 and all(isinstance(i, int) for i in kpoints)):
        raise ValueError("kpoints must be a list or tuple of three integers")

    kpoints_str = ' '.join(map(str, kpoints))
    
    content = f"""KPOINTS 
0
Gamma
{kpoints_str}
0 0 0
"""
    with open(filename, 'w') as file:
        file.write(content)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def make_incar(INCAR, crctnLst):
    #input: incar (dict) and corrections
    #return: new incar (dict)
    INCARcp = INCAR.copy()
    for crctn in crctnLst:
        for parameter, value in crctn.items():
            INCARcp[parameter] = value
    return INCARcp
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def SPINMAGMOM(atoms, INCAR):
    #SETS MAGMOM OF ATOMS 
    if 'ispin' in INCAR and INCAR['ispin'] == 2:
        ni, p, h, o = 0, 0, 0, 0
        for atom in atoms:
            if atom.symbol == 'Ni':
                ni += 1
            elif atom.symbol == 'P':
                p += 1
            elif atom.symbol == 'H':
                h += 1
            elif atom.symbol == 'O':
                o += 1
        mag = f'{ni}*2.0 {p}*1.0 {h}*1.0 {o}*1.0'
        spin_crctn = {'magmom':mag} 
        INCAR = make_incar(INCAR, [spin_crctn])
    return  INCAR 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def setupSolvolVasp(loop_lst=None, atoms=None, delta_nelect=None, flag=False):
    #delta_nelect: a list of change in num of electrons 
    #flag: '-l', '-i', '-dl', '-di'. d prefix for dimer set up   

    if (flag == '-dl' or flag == '-di'):
        incar_dct = INCARdimer 
        loop_lst = [1, 2]
        time = '4-00:00:00'
    if (flag == '-l' or flag == '-i'):
        incar_dct = INCARS
        time = '3-00:00:00'

    OUTCAR = 'OUTCAR'
    kpts_rest = [3,3,1]
    kpts_last = [6,6,1]
    nelect0, energy0 = read_nelect(OUTCAR)                
    jobFile = 'jobfile_VASPSol_voltage.sh'

    for dnel in delta_nelect:
        dir = f'dnel{dnel}'
        initPath = os.path.join('./', dir) 
        try:  
            os.mkdir(initPath)  
        except OSError:  
            pass
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(f'In directory {dir}')
        write_vasp(os.path.join(dir,'POSCAR'), atoms)
        shutil.copy('./POTCAR', f'{dir}/POTCAR')
        if (flag == '-dl' or flag == '-di'):
            shutil.copy('./MODECAR', f'{dir}/MODECAR')
        loop_str = ''
        for i in loop_lst:
            loop_str += f'{i} '
        job_find_replace = {'#SBATCH --job-name=*':f'#SBATCH --job-name={dir}',
                            '#SBATCH --time=*':f'#SBATCH --time={time}',
                            'for i in*':f'for i in {loop_str}'}
        jobScript(jobFile, dir, job_find_replace)
        for i in loop_lst:
            incext = f'vol{i}'
            kptF   = os.path.join(dir,f'KPOINTS{incext}') 
            excorr=f'in{i}'
            INCkey=f'INC{i}'
            incar, msg = incar_dct[INCkey]
            nelect = nelect0 + dnel
            vol_crctn = {'nelect':nelect}
            incar = make_incar(incar, [vol_crctn])
            incar = SPINMAGMOM(atoms, incar)
            if i==loop_lst[-1]:
                write_kpoints_file(kpts_last, kptF)
            else:
                write_kpoints_file(kpts_rest, kptF)
            write_incar(directory=dir, parameters=incar, header=incar_header)
            shutil.copy(f'{dir}/INCAR', f'{dir}/INCAR{incext}')
            print(f'Created INCAR{incext} - {msg}')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    return
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
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print(f" The NELECT before voltage calculations is {num}")
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    return nelect0, energy0
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


INCAR1 = {'gga' : 'PE',
            'pp' : 'PBE',
            'prec' : 'Accurate',
            'lreal' : 'Auto',
            'encut' : 500,
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
            'maxmove' : 0.3,
            'idipol' : 0, 
            'ldipol' : False}

msg1 = "Gas-phase coarse optimisation"

crctn1 = {'lsol' : True,
            'eb_k' : 78.4,
            'lambda_d_k' : 3.0,
            'maxmove' : 0.2,
            'istart':1,
            'icharg':1 }

msg2 = "Implicit Solvent coarse optimisation"

crctn2 = {'ediffg' : -0.025,
            'maxmove' : 0.15,
            'sigma': 0.05,
            'istart':1,
            'icharg':1 }

msg3 = "Implicit Solvent fine optimisation"

crctn3 = {'ediffg' : -0.025,
            'ispin' : 2,
            'maxmove' : 0.15,
            'sigma': 0.05,
            'istart':1,
            'icharg':1 }

msg4 = "Implicit Solvent-spin polarised fine optimisation"

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Dimer set up
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dimcrctn1 = {'ediffg': -0.06, 
            'sigma': 0.25,
            'ediff': 1e-5,
            'nsw': 200,
            'iopt': 7,
            'maxmove' : 0.15,
            'ichain': 2,
            'ddr': 0.01,
            'drotmax': 4,
            'dfnmin': 0.05,
            'dfnmax': 1.0}

msgd1 = "Implicit Solvent dimer coarse optimisation"

dimcrctn2 = {'ediffg': -0.04,
            'sigma': 0.1,
            'ediff': 1e-7,
            'nsw': 300,
            'maxmove': 0.25,
            'ddr': 5e-3,
            'dfnmin': 0.01,
            'ispin': 2}

msgd2 = "Implicit Solvent-spin polarised dimer fine optimisation"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#INCAR1                                                        #Gas-Phase coarse
INCAR2  = make_incar(INCAR1, [crctn1])                         #VASPSol coarse
INCAR3  = make_incar(INCAR1, [crctn1, crctn2])                 #VASPSol fine
INCAR4  = make_incar(INCAR1, [crctn1, crctn2, crctn3])         #VASPSol-spin fine
INCARd1 = make_incar(INCAR1, [crctn1, dimcrctn1])              #VASPSol-dimer coarse 
INCARd2 = make_incar(INCARd1, [dimcrctn2])                     #VASPSol-dimer fine

INCARS = {'INC1':[INCAR1, msg1], 'INC2':[INCAR2, msg2], 'INC3':[INCAR3, msg3], 'INC4':[INCAR4, msg4]}
INCARdimer = {'INC1':[INCARd1, msgd1], 'INC2':[INCARd2, msgd2]}
incar_header = 'INCAR created by Atomic Simulation Environment'



cliargL = len(sys.argv)
dneleAdd = []
if cliargL == 1:
    usage()
flag = sys.argv[1]

if flag in ["-i", "-l", "-dl", "-di"]:
    if ((flag == "-l" or flag == "-dl") and cliargL == 5):
        beg = float(sys.argv[2])
        end = float(sys.argv[3])
        step = float(sys.argv[4])
    elif ((flag == "-i" or flag == "-di") and cliargL >= 3):
        dneleAdd = [round(float(sys.argv[i]), 2) for i in range(2,cliargL)]
    else:
        usage()
else:
    usage()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#NELECT FOR VOLTAGE CALC 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
delta_nelect = np.array([])  
if flag == "-l" or flag == "-dl":
    lim = np.round(np.arange(beg,end,step),2) #chng in num of elect(effective charge)
    delta_nelect = np.append(delta_nelect, lim)
if flag == "-i" or flag == "-di":
    delta_nelect = np.append(delta_nelect, dneleAdd)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#ASSIGN SOLVENT-VOLTAGE CALC PARAMETERS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
loop_lst = [1, 2, 4] #choose incars from INCARS dict
Phi_SHE = 4.43 # in eV 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#FUNC INPUT PARAMETERS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
input_structure="POSCAR"
system=read(input_structure)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#CALL FUNCTIONS BELOW
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
setupSolvolVasp(loop_lst, system, delta_nelect, flag)
