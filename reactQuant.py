#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import re
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from matplotlib import font_manager
from customFunc import formatImg, CustomPrinter 

e_in_C = 1.602176634e-19 #C
#pwd = os.path.basename(os.getcwd())

cliargL = len(sys.argv)
if cliargL > 1:
    flag = sys.argv[1]
else:
    flag = "-n"

if flag == "-e":
    states = {f'{sys.argv[i]}': sys.argv[i] for i in range(2,cliargL)}
elif flag == "-r":
    froot = True
    if sys.argv[2]:
        guess = float(sys.argv[2])
    else:
        guess = 0
    flag = "-n"
    states = {'IS': 'initial', 'FS': 'final'} 
elif flag == "-n":
    froot = False
    states = {'IS': 'initial', 'FS': 'final'} 
    

def parse_output(parsefile, fnName):
    #parsefile (fileobj): file to parse
    #fnName (str): function name for parms search

    with open(parsefile, 'r') as file:
        data = file.read()
    
    Phi_SHE = re.search('Phi_SHE.*',data).group().split()[-1]
    Phi_SHE = float(Phi_SHE)
    pattern = fr"(?<=Fitted Function: {fnName}\n)(.*\n)*?^Capacitance.*"
    matches = re.search(pattern, data, re.MULTILINE)
    if matches:
        output = matches.group().strip()

    param_pattern = r"Parameter 1: ([\d\.\-e]+)\s+Parameter 2: ([\d\.\-e]+)\s+Parameter 3: ([\d\.\-e]+)"
    params_match = re.search(param_pattern, output, re.DOTALL)
    a0, a1, a2 = map(float, params_match.groups()) \
                if params_match else (None, None, None)

    cov_matrix_pattern = r"Covariance Matrix:\n((?:\s+[\d\.\-e]+\s+[\d\.\-e]+\s+[\d\.\-e]+\n){3})"
    cov_match = re.search(cov_matrix_pattern, output)
    covariance_matrix = np.array([list(map(float, line.split())) \
                        for line in cov_match.group(1).strip().split("\n")]) \
                        if cov_match else None

    capacitance_pattern = r"Capacitance = ([\d\.\-e]+)"
    cap_match = re.search(capacitance_pattern, output)
    C = float(cap_match.group(1)) if cap_match else None

    return [a0, a1, a2, covariance_matrix, C, Phi_SHE]

def fitFn(x, a0, a1, a2):
    return a0 + a1 * x + a2 * (x ** 2)

def Er(x, pIS, pFS):
    yIS = fitFn(x, pIS[0], pIS[1], pIS[2])
    yFS = fitFn(x, pFS[0], pFS[1], pFS[2]) 
    er = yFS - yIS
    return er 


def xyrootEr(pIS, pFS, guess):
    root = fsolve(Er, x0=guess+pIS[5], args=(pIS, pFS))
    xrt, yrt = root[0]- pIS[5], Er(root, pIS, pFS)
    return  xrt, yrt

def volOutDict(volDatF):
    #volDatF: vol data file path 
    df = pd.read_csv(volDatF, delimiter=' ')
    return df

def get_xlims(df):
    dfmin = df['#WrkFnc'].min() 
    dfmax = df['#WrkFnc'].max() 
    return dfmin, dfmax

def collectISFS(states):
    outF, optParamsF = 'VOLTAGE-OUTPUT.dat', 'voltage-analysis.dat' 
    #states = {'IS': 'initial', 'FS': 'final'} 
    dfDict = {}
    xlims = {'xmin': [], 'xmax': []} 
    func = 'fitFn'
    parmsDict = {}
    xminDict, xmaxDict = {}, {}
    xdict, ydict, Cdict = {}, {}, {}
    ny, nEr = 100, 200
    for st in states:
        dir = states[st]
        volDatF, parmF = os.path.join(dir, outF), os.path.join(dir, optParamsF)
        df = volOutDict(volDatF)
        dfDict[st] = df
 
        xmin, xmax = get_xlims(df)
        xdict[st] =  np.linspace(xmin, xmax, ny) 

        xlims['xmin'].append(xmin), xlims['xmax'].append(xmax)
        parmsDict[st] = parse_output(parmF, func) #[a0, a1, a2, cov_mat, Cap, Phi_SHE] 

    xtrMin = min(xlims['xmin']) 
    xtrMax = max(xlims['xmax']) 

    xlims['xmin'] = max(xlims['xmin']) 
    xlims['xmax'] = min(xlims['xmax'])

    for st in states:
        p = parmsDict[st]
        y = fitFn(xdict[st], p[0], p[1], p[2]) 
        ydict[st], Cdict[st] = y, p[4]

        xsctr = dfDict[st]['#UvSHE']
        ysctr = dfDict[st]['#Omega']
        plt.plot(xdict[st]-p[5],ydict[st], label = f'{st}',zorder=1) 
        plt.scatter(xsctr, ysctr, edgecolor='black', marker='o', s=80,zorder=2)
    
    imgname = "".join(states.values()) 
    formatImg(sv=f'{imgname}_Ene.png', xl='UvSHE (V)', yl='Energy (eV)')
    cP = CustomPrinter(f'{imgname}.dat')

    if flag == "-n":
        x = np.linspace(xtrMin,xtrMax, nEr)
        pIS, pFS = parmsDict['IS'], parmsDict['FS']

        dC = np.round(pFS[4] - pIS[4],2)
        dC_in_F = e_in_C * dC #Farad
        cP.cp(f'The reaction Capacitance change = {dC} e/V')
        cP.cp(f'The reaction Capacitance change = {dC_in_F:.2e} F')
         
        yEr = Er(x, pIS, pFS)

        plt.plot(x-pIS[5], yEr, label=r'$\text{E}_{\text{R}}$')
        for x in xlims.values():
            plt.axvline(x=x-pIS[5], color='black', linestyle='--')
        plt.axhline(y=0, color='black', linestyle='--')
        if froot:
            xrtEr, yrtEr = xyrootEr(pIS, pFS, guess)
            plt.plot(xrtEr, yrtEr, marker='o',label=f'({xrtEr:.2}, {yrtEr[0]:.2})')
            cP.cp(f'UvSHE of zero Er = {xrtEr:.2} V')
   
        formatImg(sv='React_Ene.png', xl='UvSHE (V)', yl='Energy (eV)',text=fr'$\Delta\text{{C}}_\text{{R}}$ = {dC_in_F:.2e} F')


#def formatImg(sv='plot.png',xl='xaxis',yl='yaxis',text=None):
#    w = 545
#    wd = 1.5
#    fig, ax = plt.gcf(), plt.gca()
#
#    font_prop = font_manager.FontProperties(weight=1000)
#
#    for spine in ax.spines.values():
#        spine.set_linewidth(wd)
#
#    for line in ax.lines:
#        line.set_linewidth(wd)
#
#    ax.tick_params(axis='both', which='major', labelsize=12, width=wd)
#    ax.tick_params(axis='both', which='minor', labelsize=10, width=wd)
#    
#    ax.set_xlabel(xl, fontsize=12, fontweight=w)
#    ax.set_ylabel(yl, fontsize=12, fontweight=w)
#
#    plt.xticks(fontsize=12, fontweight=w)  
#    plt.yticks(fontsize=12, fontweight=w) 
#    
#    plt.tight_layout()
#
#    legend = plt.legend(prop=font_prop)
#    if text:
#        bbox = legend.get_window_extent(fig.canvas.get_renderer())
#        bbox_data = ax.transData.inverted().transform(bbox)
#
#        text_x = bbox_data[1][0] + 0.3  #(bbox_data[0][0] + bbox_data[1][0]) / 2  
#        text_y = bbox_data[1][1] - 0.025 
#       
#        lgnd_fsz = legend.get_texts()[0].get_fontsize()
# 
#        ax.text(
#            text_x, text_y,
#            text,
#            fontsize=lgnd_fsz,
#            va='top',
#            ha='center'
#        )
#       
#    plt.savefig(f'{pwd}-{sv}')
#    plt.close()

if __name__ == "__main__":
    collectISFS(states)

    
