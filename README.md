# Scripts  
Python and Bash scripts to run and post-process VASP calculations.  

### `setSolVol.py`  

For geometry optimisations with varying N electrons (voltage scan) use options `-l` or `-i`. For Dimer calculations with varying N electrons (voltage scan) use options `-dl` or `-di`.
 


#### Usage: Option -l/-dl defines limits of dnelect directories to be made. For dnelect values from -1.5 to 1.6 in steps of 0.3, use the below command. 
```bash
setSolVol.py -l/-dl -1.5 1.6 0.3
```


#### Usage: Option -i/-di creates directories of individual dnelect. For dnelect values 1.0, -1.0, -1.2 , use the below command.
```bash
setSolVol.py -i/-di 1.0 -1.0 -1.2
```
The calculations will request 3 (4) days for geometry optimisation (Dimer calculation). If need to change, find it in setSolVol.py by, 

```bash
grep "time = " setSolVol.py
```
