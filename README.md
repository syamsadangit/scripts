# Scripts  
Python and Bash scripts to run and post-process VASP calculations.  

### `setSolVol.py`  

For geometry optimisations with varying N electrons (voltage scan) use options -l or -i
For Dimer calculations with varying N electrons (voltage scan) use options -dl or -di
 
option -l/-dl for limits. Start, end and step of dnelect directories to be made.

####Usage: 
```bash
setSolVol.py -l/-dl -1.5 1.6 0.3

option -i/-di creates directories of dnelect-values 1.0 -1.0 -1.2

####Usage: 
```bash
setSolVol.py -i/-di 1.0 -1.0 -1.2

The calculations will request 3 (4) days for geometry optimisation (Dimer calculation). If need to change, change in setSolVol.py 
