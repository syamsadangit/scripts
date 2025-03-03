#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ase.calculators.cp2k import CP2K
from ase.io.cp2k import read_cp2k_restart as getAtoms   
from ase.atoms import Atoms
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.optimize import FIRE
import os, re, sys, subprocess, shutil
import numpy as np
from ase.geometry.geometry import get_layers
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt
from matplotlib import font_manager
pwd = os.path.basename(os.getcwd())



# Define class
#-----------------------------------------------
class runCP2K:

    def __init__(self, cp2k_cmd=None, label=None, makeInp=False, isIFile=False, 
                IFile=None, inp=None, strucIFile=None, strucOFile=None, atoms=None, restart=False):
        self.cp2k_cmd = cp2k_cmd
        if self.cp2k_cmd:
            proceedCalc = True
        else:
            proceedCalc = False
        self.makeInp = makeInp #To make input file from cp2k input parameters in a file and structure file (POSCAR)
        self.isIFile = isIFile
        self.IFile = IFile  #read file with name IFile
        self.inp = inp #str with cp2k parms
        self.strucIFile = strucIFile  #Read structue file (eg. POSCAR) to Atoms object atoms. fmt auto detect
        self.strucOFile = strucOFile  #Write structue file (eg.CONTCAR) from Atoms object atoms. fmt auto detect
        if label:             
            self.prjN = label
        else:
           self.prjN = 'CP2Kcalc'
        self.atoms = atoms
        if self.strucIFile:
            self.atoms = read(self.strucIFile)
        if isinstance(atoms, Atoms):
            self.atoms = atoms
        self.restart = restart

        #Write out info
        #-----------------------------------------------
        self.cP = CustomPrinter("calcINFO.out")
        self.instVar = '\n'.join([f"{key} = {val}" for key, val in self.__dict__.items()])
        self.cP.cp("-------------------------------------------------")  
        self.cP.cp("-----------------Input Params--------------------")  
        self.cP.cp(self.instVar)
        self.cP.cp("-------------------------------------------------")  
        self.save_filtered_env_vars("my_env_vars.txt")
        #-----------------------------------------------
        
        
        if self.restart:
            fname = self.IFile.split("-1.restart")[0]
            finStrucF = self.IFile 
            self.OFile = f'{fname}.out'
        else:
            if self.isIFile:
                if self.makeInp:  
                    with open(self.IFile, 'r') as IF: 
                        self.inp = IF.read()
                    self.mkCP2KinpF(inp=self.inp, atoms=self.atoms, prjN=self.prjN)  #create .inp file using parms and atoms
                    finStrucF = f'{self.prjN}-1.restart'
                    self.OFile = f'{self.prjN}.out'
                elif self.IFile:
                    if self.IFile.endswith(".inp"):
                        fname = self.IFile.split(".inp")[0]
                        self.ioMod(IFile, 'PROJECT.*\n', f'PROJECT {fname}\n')
                        finStrucF = f'{fname}-1.restart'
                        self.OFile = f'{fname}.out' 
                    elif self.IFile.endswith("-1.restart"):
                        fname = self.IFile.split("-1.restart")[0]
                        self.ioMod(IFile, 'PROJECT.*\n', f'PROJECT {fname}\n')
                        finStrucF = self.IFile
                        self.OFile = f'{fname}.out'
                else:
                    self.cP.cp(f'Warning: isIFile is {self.isIFile} but makeInp is {self.makeInp} and IFile is {self.IFile}.') 
                    proceedCalc = False
            elif self.inp:
                self.mkCP2KinpF(inp=self.inp, atoms=self.atoms, prjN=self.prjN) #create .inp file using parms and atoms
                finStrucF = f'{self.prjN}-1.restart'
                self.OFile = f'{self.prjN}.out'
            else:
                self.cP.cp(f'Warning: isIFile is {self.isIFile} and inp is {self.inp}.') 
                proceedCalc = False

        if proceedCalc: 
            #----------------Calculate----------------------
            #self.calculate(cp2k_cmd=self.cp2k_cmd, restart=self.restart)
            self.cP.cp('CP2K calculator is setup. Use obj.calculate() to run the calculation.')
        else:
            self.cP.cp('CP2K calculator setup failed. Verify the combination of parameters used.')

    def calculate(self):
        restart = chkbool(self.restart)
        
        #Calculate
        #-----------------------------------------------
        if restart:
            self.cP.cp("-------------------------------------------------")
            self.cP.cp("Restarting the calculation.")
            self.cP.cp("-------------------------------------------------")
        os.system(self.cp2k_cmd) 
        #----------------Collect Atoms------------------
        self.rw_cp2k_atoms(finStrucF=finStrucF, strucOFile=self.strucOFile) 
        #----------------Collect Quantities-------------
        res = self.results(OFile=self.OFile)
        self.cP.cp("---------------Calculation Results---------------")
        self.cP.cp(f'{res}')
        self.cP.cp("-------------------------------------------------")



    def mkCP2KinpF(self, inp=None, atoms=None, prjN=None): 

        inp = self.ioMod(inp, 'PROJECT.*\n', f'PROJECT {prjN}\n')    
    
        cstmParms = dict(
                    auto_write=True,
                    basis_set=None,
                    charge=0,
                    cutoff=None, #set in inp 
                    force_eval_method="Quickstep",
                    inp=inp,
                    max_scf=None,
                    multiplicity=None,
                    pseudo_potential=None,
                    stress_tensor=True,
                    uks=False,
                    poisson_solver='auto',
                    xc='PBE',
                    print_level=None,
                    set_pos_file=False) 
         

        ASE_CP2K_COMMAND = os.environ.get("ASE_CP2K_COMMAND")
        self.cP.cp(f'ase cp2k command env var is "{ASE_CP2K_COMMAND}"')
        with CP2K() as calc:
            self.cP.cp(calc.command)
            calc.label = prjN
            cnst = atoms.constraints
            if cnst:
                constAtoms = '''
                                &CONSTRAINT
                                    &FIXED_ATOMS
                                        COMPONENTS_TO_FIX XYZ
                                        LIST 1
                                    &END FIXED_ATOMS
                                &END CONSTRAINT
                            '''
                c =  cnst[0] 
                cnstInd = ' '.join([str(ind + 1) for ind in c.get_indices()])
                fin, rep = r'LIST.*\n', fr'LIST {cnstInd}\n' 
                addText1 = re.sub(fin, rep, constAtoms)
                cstmParms['inp'] = re.sub(fr"(?mi)(?=.*END GEO_OPT)(.*?$)",fr"\1\n{addText1}",cstmParms['inp']) 
            calc.set(**cstmParms)
            calc.atoms = atoms
            calc._create_force_env()  

    def rw_cp2k_atoms(self, finStrucF=None, strucOFile=None):

        with open(finStrucF,'r') as fr:
            NewAtoms = getAtoms(fr)
            fr.seek(0)
            lines = fr.read()

        cnstBlk = re.search(r'&FIXED_ATOMS(.*?)&END FIXED_ATOMS', lines, re.S).group(1)

        ind = []
        for l in re.findall(r'(\d+\.\.\d+|\d+)', cnstBlk):
            if '..' in l:
                st, en = map(int,l.split('..'))
                asest, aseen = st-1, en-1
                ind.extend(range(asest,aseen+1))
            else:
                aseelm = int(l) - 1
                ind.append(aseelm)
        #self.cP.cp("-------------------------------------------------")  
        #self.cP.cp("-------------ASE constraint index----------------")  
        #self.cP.cp(f"{ind}")  
        #self.cP.cp("-------------------------------------------------")  

        cnst = FixAtoms(indices=ind)
        NewAtoms.set_constraint(cnst)

        self.cP.cp("-------------------------------------------------")  
        self.cP.cp(f'out - {NewAtoms}')
        self.cP.cp("-------------------------------------------------")  

        with open(strucOFile,'w') as fw:
            write(fw,NewAtoms) 

        return 

    def results(self, OFile=None):
        res = {'energy':[], 'force':[]}
        enPat = 'ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]:'
        enKey = 'Total Energy'
        fKey = 'Max. step size'
        LastOptInfo = {}
        if os.path.isfile(OFile):
            with open(OFile, 'r') as of:
                lines = of.readlines()
            revlines = list(reversed(lines))
            for ind, L in enumerate(revlines, start=0):
                if 'Informations' in L:
                    imax, imin = ind, ind-20
                    break
            for l in revlines:
                if enPat in l:
                    en = l.strip(enPat)
                    break
            revRange = reversed(range(imin, imax+1))
            for i in revRange:
                lastInfo = revlines[i].split("=")
                keyval = []
                for elm in lastInfo:
                    keyval.append(elm.strip(" \n-"))
                if len(keyval) > 1:
                    LastOptInfo[f'{keyval[0]}'] = keyval[1]
        LastOptInfo[enKey] = en
        for key, val in LastOptInfo.items():
            try:
                float(val)
                LastOptInfo[key] = float(val)
            except ValueError:
                pass
        res['energy'], res['force'] = [LastOptInfo[enKey]], [LastOptInfo[fKey]]
        return res
            
 
    def ioMod(self, IForstr, finStr, repStr, OForstr=None):
        """
            Modify file content or a string by replacing occurrences of `finStr` with `repStr`.

            Parameters:
            - IForstr (str): File path or a string to process.
            - finStr (str): Pattern to search for (supports regex).
            - repStr (str): Replacement string.
            - OForstr (Optional[str]): Output file path. If None, modifies `IForstr` in place.

            Behavior:
            - If `IForstr` is a file, reads and modifies its content.
            - Writes modified content to `OForstr` (if provided) or back to `IForstr`.
            - If `OForstr` exists, the program exits with an error.
            - If `IForstr` is a string, returns the modified string directly.

            Returns:
            - Modified string if `IForstr` is not a file.
        """
        if os.path.isfile(IForstr):
            #if OForstr: 
            #pass
            #if ( not os.path.isfile(OForstr)):        
            #    sys.exit(1)
            if not OForstr:
                OForstr = IForstr
            with open(IForstr, 'r') as f:
                lines = f.readlines() 
            with open(OForstr, 'w') as f:
                for line in lines:
                    nline = re.sub(finStr, repStr, line)
                    f.write(nline)
        elif isinstance(IForstr, str): 
            OForstr = re.sub(finStr, repStr, IForstr)
            return OForstr 

    def save_filtered_env_vars(self, filename="environment_variables.txt"):
        """
        Filters environment variables containing 'SLURM', 'SBATCH', 'OMP', or 'MPI',
        sorts them alphabetically, and saves them to a specified file.

        Args:
            filename (str): Name of the output file to save the filtered variables.

        Returns:
            None
        """
        # Define keywords to filter by
        keywords = ['SLURM', 'SBATCH', 'OMP', 'MPI']
        
        # Filter environmental variables based on keywords
        filtered_vars = {key: value for key, value in os.environ.items() if any(keyword in key for keyword in keywords)}
        
        # Sort the variables alphabetically
        sorted_vars = dict(sorted(filtered_vars.items()))
        
        # Write to file
        with open(filename, "w") as f:
            f.write("Environment Variables Containing SLURM, SBATCH, OMP, or MPI:\n")
            f.write("=" * 60 + "\n")
            for key, value in sorted_vars.items():
                f.write(f"{key:<30} : {value}\n")
        
        print(f"Filtered environment variables have been saved to '{filename}'.")


class CustomPrinter:
    def __init__(self, filename="customInfo.out"):
        """
        Initializes the custom printer to write all output to a specified file.
        Args:
            filename (str): The file to write the output.
        """
        self.filename = filename
        # Clear the file if it already exists
        with open(self.filename, "w") as f:
            f.write("")  # Clear file content

    def cp(self, *args, sep=" ", end="\n"):
        """
        Writes the given output to the specified file in the same format as print().
        
        Args:
            *args: Values to be printed, separated by `sep` and terminated with `end`.
            sep (str): Separator between values (default: " ").
            end (str): End of the line (default: "\n").
        """
        output = sep.join(map(str, args)) + end
        with open(self.filename, "a") as f:
            f.write(output)


class Dope():
    def __init__(self, strucF=None):
        self.strucF = strucF
        self.slab = read(strucF)
        self.cP = CustomPrinter('dopingReport.out')

    def addAtoms(self, dopant=None , modLay=None, modSpecies=None, modInd=None, modPos=None, miller=(0,0,1), tolerance=0.15):
        modslab = self.slab.copy()
        indL, distL = get_layers(self.slab,miller,tolerance)
        AtSym = self.slab.get_chemical_symbols()
        #print(AtSym)
        AtPos = self.slab.get_positions()
        AtInd = [ atom.index for atom in self.slab]
        L, nL, species = set(indL), len(distL), set(AtSym)
        infoDf = pd.DataFrame({'Layers': indL, 'AtomSymbol': AtSym,
                'AtomIndex': AtInd, 'AtomX': AtPos[:,0],
                'AtomY': AtPos[:,1], 'AtomZ': AtPos[:,2], })

        lspn = [(l, sp, len(infoDf[(infoDf['Layers'] == l) & \
                (infoDf['AtomSymbol'] == sp)])) for l in L \
                for sp in species]

        self.cP.cp('_'*45)
        self.cP.cp('If the below info is not expected,\ncheck "miller" and "tolerance" in pickLayer()')
        self.cP.cp('Layer Species Natoms')
        self.cP.cp('-'*45)
        for elm in lspn:
            self.cP.cp(f'    {elm[0]}       {elm[1]}      {elm[2]}')
        self.cP.cp('_'*45,'\n')

        modDf = infoDf.copy()
        modAtSym = AtSym
        if modLay:
            modDf = modDf[modDf['Layers'].isin(modLay)]
        if modSpecies:
            modDf = modDf[modDf['AtomSymbol'].isin(modSpecies)]
        if modInd:
            modDf = modDf[modDf['AtomIndex'].isin(modInd)]
            if dopant:
                for i, ind in enumerate(modInd):
                    modAtSym[ind] = dopant[i]
        if modPos:
            modPos = [tuple(pos) if isinstance(pos, (list, np.ndarray)) else pos for pos in modPos]
            modDf = modDf[
                modDf.apply(lambda row: (row['AtomX'], row['AtomY'], row['AtomZ']) in modPos, axis=1)
            ]
        #print(modDf)
        modslab.set_chemical_symbols(modAtSym)
        resDict = {'numlay': nL, 'lay': L, 'distlay': distL}
        return modslab



# Define functions
#-----------------------------------------------
def TestConv(optParm,find,series,restart=False,prjN=None,cellOpt=False):

    #Example variable def 
    #optParm = 'CUTOFF'
    #find = fr"(?<!REL_){optParm}.*\n" 
    #series = np.arange(450, 1200, 50) 

    #if convTest:
    for value in series:
        print("-------------------------------------------------")
        print(f'Running {optParm} {value} ...')
        replace = f"{optParm} {value}\n"
        inpN = re.sub(find, replace, inp)
        if chkbool(restart):
            rstF = f'{prjN}-1.restart'
            if(rstF not in os.listdir('.')):
                rstF = f'{prjN}.inp'
                restart = False
            with open(rstF, 'r') as F:
                rstFstr = F.read() 
            rstFstr = re.sub(find, replace, rstFstr)
            with open(rstF, 'w') as F:
                F.write(rstFstr)
        #cp2k(inpN,restart,prjN)
        restart = False
        optDir = f"{value}_{optParm}"
        if not os.path.exists(optDir):
            os.makedirs(optDir)
            print(f"Directory {optDir} created.")
        for file in os.listdir('.'):
            if (os.path.isfile(file) and
                "restart" not in file.lower() and
                not file.endswith(".py") and
                not file.endswith(".inp") and
                file != 'POSCAR'):
                shutil.move(file, os.path.join(optDir, file))
                print(f"Moved {file} to {optDir}.")
            if (os.path.isfile(file) and
                (file.endswith(".inp") or file == 'POSCAR')):
                shutil.copy(file, os.path.join(optDir, file))
                print(f"Copied {file} to {optDir}.")
        print(f'Done {optParm} {value}')
        print("-------------------------------------------------")


def chkbool(v):
    if type(v).__name__ == 'str':
        val = v.lower() in ("yes", "true", "t", "1")
    elif type(v).__name__ == 'bool':
        val = v
    elif type(v).__name__ == 'int':
        val = v in (1)
    else:
        val = False
    return val 

def mkdir(dirname):
    try:
        os.mkdir(dirname)
        print('_'*45,'\n')
        print(f"Directory '{dirname}' created.")
    except FileExistsError:
        print('_'*45,'\n')
        print(f"Directory '{dirname}' already exists.")
    except Exception as e:
        print('_'*45,'\n')
        print(f"An error occurred: {e}") 
    return


def formatImg(sv='plot.png',xl='xaxis',yl='yaxis',text=None):
    w = 545
    wd = 1.5
    fig, ax = plt.gcf(), plt.gca()

    font_prop = font_manager.FontProperties(weight=1000)

    for spine in ax.spines.values():
        spine.set_linewidth(wd)

    for line in ax.lines:
        line.set_linewidth(wd)

    ax.tick_params(axis='both', which='major', labelsize=12, width=wd)
    ax.tick_params(axis='both', which='minor', labelsize=10, width=wd)
    
    ax.set_xlabel(xl, fontsize=12, fontweight=w)
    ax.set_ylabel(yl, fontsize=12, fontweight=w)

    plt.xticks(fontsize=12, fontweight=w)  
    plt.yticks(fontsize=12, fontweight=w) 
    
    plt.tight_layout()

    legend = plt.legend(prop=font_prop)
    if text:
        bbox = legend.get_window_extent(fig.canvas.get_renderer())
        bbox_data = ax.transData.inverted().transform(bbox)

        text_x = bbox_data[1][0] + 0.3  #(bbox_data[0][0] + bbox_data[1][0]) / 2  
        text_y = bbox_data[1][1] - 0.025 
       
        lgnd_fsz = legend.get_texts()[0].get_fontsize()
 
        ax.text(
            text_x, text_y,
            text,
            fontsize=lgnd_fsz,
            va='top',
            ha='center'
        )
       
    plt.savefig(f'{pwd}-{sv}')
    plt.close()


# Define function to compute all the metrics
def evaluate_predictions(measuredData, trueData):
    """
    Compute multiple error metrics comparing measured data to true data.
    
    Parameters:
    measuredData (array-like, pd.Series, or list): Predicted or measured values (e.g., MACE results).
    trueData (array-like, pd.Series, or list): Ground truth values (e.g., VASP results).
    
    Returns:
    dict: A dictionary containing all the computed metrics.
    """
    # Convert inputs to numpy arrays for easier computation
    if isinstance(measuredData, (pd.Series, list)):
        measuredData = np.array(measuredData)
    if isinstance(trueData, (pd.Series, list)):
        trueData = np.array(trueData)
    
    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(trueData, measuredData)
    
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(trueData, measuredData))
    
    # Mean Absolute Percentage Error (MAPE) - Avoid division by zero
    mape = np.mean(np.abs((measuredData - trueData) / trueData)) * 100
    
    # R-squared (R²)
    r2 = r2_score(trueData, measuredData)
    
    # Pearson Correlation Coefficient (r)
    correlation_matrix = np.corrcoef(measuredData, trueData)
    pearson_r = correlation_matrix[0, 1]
    
    # Bias (Mean Error)
    bias = np.mean(measuredData - trueData)
    # Standard Deviation of Error (σ)
    std_dev_error = np.sqrt(np.mean((measuredData - trueData - bias) ** 2))
    
    
    # Store results in a dictionary
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "Pearson_r": pearson_r,
        "Std_Dev_Error": std_dev_error,
        "Bias": bias
    }
    
    return metrics
