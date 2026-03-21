#!/usr/bin/env python3
from __future__ import annotations 
import argparse 
from typing import Iterable, List, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path 
import math

class FitShomate:
    def __init__(
            self, 
            R: float = 1.98720425864083,
            T_col: str = 'T',
            Cp_col: str = 'Cp'): 
        self.R = R 
        self.T_col = T_col 
        self.Cp_col = Cp_col

    def read(self, path: Path) -> pd.DataFrame: 
        # Use pandas' Python parsing enginge
        df = pd.read_csv(path, sep=r"\s+", engine="python")
        out = df[[self.T_col, self.Cp_col]].copy() 
        out[self.T_col] = pd.to_numeric(out[self.T_col], errors="coerce") 
        out[self.Cp_col] = pd.to_numeric(out[self.Cp_col], errors="coerce") 
        out = out.dropna() 
        return out

#    def fit(self, df: pd.DataFrame) -> Dict[str, float]:
#        T = df[self.T_col].to_numpy(dtype=float)
#        Cp = df[self.Cp_col].to_numpy(dtype=float)
#        # Perform fit
##        Cp_red = Cp / self.R # y 
##        sh_coeffs = np.polyfit(T, Cp_red, int(4))
#        A = np.column_stack([np.ones_like(T), T, T**2, T**3, T**4]) # creates an array of ones with the same shape as T; takes each 1D array and stacks them as columns 
#        # Wrap coefs into a poly fn 
##        sh_poly = np.poly1d(sh_coeffs)  
##        Cp_red_fit = sh_poly(T)  
##        sh_coeffs_asc = sh_coeffs[::-1] 
#        Cp_fit_coeffs = np.linalg.lstsq(A, Cp, rcond=None)[0]
#        result = {f"a{i}": float(c) for i, c in enumerate(Cp_fit_coeffs)} 
#        return result 
#
    def fit(self, df:pd.DataFrame) -> Dict[str, float]: 

        T = df[self.T_col].to_numpy(dtype=float)
        Cp = df[self.Cp_col].to_numpy(dtype=float)
        Tb = int(1000) # split T 

        down = T <= Tb
        up = T >= Tb 

        T_low, Cp_low = T[low], Cp[low]
        T_high, Cp_high = T[high], Cp[high]

        A_low = np.column_stack([np.ones_like(T[T_low]), T[T_low], T[T_low]**2, T[T_low]**3, T[T_low]**4]) # creates an array of ones with the same shape as T; takes each 1D array and stacks them as columns 
        A_high = np.column_stack([np.ones_like(T[T_high]), T[T_high], T[T_high]**2, T[T_high]**3, T[T_high]**4]) # creates an array of ones with the same shape as T; takes each 1D array and stacks them as columns 

        coef_low = np.linalg.lstsq(A_low, Cp_low, rcond=None)[0]
        coef_high = np.linalg.lstsq(A_high, Cp_high, rcond=None)[0]
    
    def Hpoly(Tx, coef):
        """Integrate Cp quartic""" 
        a0, a1, a2, a3, a4 = coef  
        F_x = a1*Tx + 0.5*a2*Tx**2 + (a3/3.0)*Tx**3 + (0.25)*a4*Tx**4 + 0.2*a5*Tx**5 
        return f"{F_x:,8e}"

    def name(self, path: Path) -> str:
        name = path.stem 
        for suffix in ("nasa7", "_2000", "_fullT"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        return name

#    def tocsv(self, path: Path, suffix: str = "_fit.csv") -> str: 
#        return f"{path.stem}{suffix}"

    def fit_one(self, path: Path) -> Dict[str, float]:
        df = self.read(path) # read() imports a path, exports df  
        fit = self.fit(df) # fit() imports a df, exports a dict  
        fit["Molecule"] = self.name(path) # name imports a path, outputs a str
        return fit 

    def fit_all(self, paths: Iterable[Path]) -> pd.DataFrame: 
        rows: List[Dict[str, float]] = []
        for p in paths:
            rows.append(self.fit_one(p)) 
        return pd.DataFrame(rows) 
    
def collect(inps: List[str], recursive: bool) -> List[Path]: 
    files: List[Path] = []
    for p in inps: 
        path = Path(p) 
        if path.is_file():
            files.append(path) 
        elif path.is_dir():
            pattern = "**/*_nasa7.txt" if recursive else "*_nasa7.txt" 
            files.extend(sorted(path.glob(pattern))) 
        else: 
            files.extend(sorted(Path(".").glob(p))) 

    # deduplicate 
    seen = set() 
    unique = []
    for f in files:
        rf = f.resolve() 
        if rf not in seen:
            seen.add(rf) 
            unique.append(f) 

    return unique 
    
def main() -> None:
    parser = argparse.ArgumentParser() 
    parser.add_argument(
        "inputs",
        nargs="+"
    ) 
    parser.add_argument("--out", default="nasa7_fit_results.csv", help="Output CSV path") 
    parser.add_argument("--recursive", action="store_true") 
    parser.add_argument("--T-col", default="T", help="Temperature column label")
    parser.add_argument("--Cp-col", default="Cp", help="Heat Capacity column label") 
    parser.add_argument("--R", type=float, default=1.98720425864083, help="Gas constant for Cp/R")  
    args = parser.parse_args() 
    paths = collect(args.inputs, args.recursive) # (args.inputs = "file1 file2 .. filen.txt" | args.recursive = positional argument that is required for the collect() function)  

    if not paths: 
        raise SystemExit("No input file provided") 

    fitter = FitShomate(
        R = args.R,
        T_col = args.T_col,
        Cp_col = args.Cp_col
    ) 
    
    results = fitter.fit_all(paths)
    coeffs = [c for c in results.columns if c.startswith("a")] 
    lead = ["Molecule"] + sorted(coeffs, key=lambda x: int(x[1:])) # drops first char and sorts by int
    tail = [c for c in results.columns if c not in lead] 
    results = results[lead + tail]
    results.to_csv(args.out, index=False)
    print(f"results length: {len(results)}") 

if __name__ == "__main__":
    main() 


