#!/usr/bin/env python3
from __future__ import annotations 
import argparse 
from typing import Iterable, List, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path 
import math

class FitNASA:
    def __init__(
            self, 
            R: float = 1.98720425864083,
            R_CAL = 1.98720425864083,  # cal/mol-K
            HARTREE_TO_KCAL = 627.5094740631,
            T_col: str = 'T',
            Cp_col: str = 'Cp'): 
        self.R = R 
        self.HRT = HARTREE_TO_KCAL
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

    def fit(self, df:pd.DataFrame) -> Dict[str, float]: 
        T = df[self.T_col].to_numpy(dtype=float)
        Cp = df[self.Cp_col].to_numpy(dtype=float)
        Tb1 = int(200)
        cut = T >= Tb1 # postulate: cut represents indices 
        T_cut, Cp_cut = T[cut], Cp[cut]
        A_cut = np.column_stack([np.ones_like(T_cut), T_cut, T_cut**2, T_cut**3, T_cut**4])
        coeff_cut = np.linalg.lstsq(A_cut, Cp_cut, rcond=None)[0]
        result = {f"a{i}": float(c) for i, c in enumerate(coeff_cut)}
        return result 

    def name(self, path: Path) -> str:
        name = path.stem 
        for suffix in ("nasa7", "_2000", "_fullT"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        return name

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

    fitter = FitNASA(
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


