# scripts
Python and Bash scripts to run and post process VASP calculations. 

Usage

reactQuant.py

<u>setSolVol.py</u>
For geometry optimisations with varying N electrons (voltage scan) use options -l or -i
For Dimer calculations with varying N electrons (voltage scan) use options -dl or -di
 
Usage: setSolVol.py -l/-dl start stop step
    option -l/-dl for limits. Start, end and step of dnelect dir to be made.
    eg. setSolVol.py -l/-dl -1.5 1.6 0.3

Usage: setSolVol.py -i/-di dnelect-values
    option -i/-di for include individual dnelect-values
    eg. setSolVol.py -i/-di 1.0 -1.0 -1.2

The calculations will request 3 (4) days for geometry optimisation (Dimer calculation). If need to change, change in setSolVol.py 

