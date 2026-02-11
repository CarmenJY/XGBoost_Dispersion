#!/usr/bin/env python
"""
C4 Baseline Model for Zn²⁺–(H₂O)n Induction Energy

This script implements a physics-based baseline for ion-dipole induction:
    E_baseline = -C₄/R⁴

Inspired by the SAPT-D3 implementation (saptd3.py) but adapted for 
induction/polarization in Zn²⁺-water clusters.

Usage as script:
    python baseline_c4.py cluster.xyz --kcal

Usage as module:
    from baseline_c4 import C4Baseline
    baseline = C4Baseline(mol, C4=1.0, kcal=True)
    print(baseline.e_ind)  # total baseline energy
"""

import math
import sys
from argparse import ArgumentParser
from typing import List, Tuple, Dict, Optional

# Conversion factors
ANGSTROM_TO_BOHR = 1.0 / 0.5291772083
BOHR_TO_ANGSTROM = 0.5291772083
HARTREE_TO_KCAL = 627.5095


def read_xyz(xyz_path: str) -> Dict:
    """
    Read XYZ file and return molecular data.
    
    :param xyz_path: Path to XYZ file
    :return: Dictionary with 'index', 'atom', 'coordinates' lists
    """
    mol = {'index': [], 'atom': [], 'coordinates': []}
    
    with open(xyz_path, 'r') as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    
    if len(lines) < 3:
        raise ValueError("XYZ file too short")
    
    try:
        nat = int(lines[0])
    except ValueError as e:
        raise ValueError("First line must be number of atoms") from e
    
    coord_lines = lines[2:2 + nat]
    if len(coord_lines) != nat:
        raise ValueError(f"Expected {nat} coordinate lines, got {len(coord_lines)}")
    
    for i, line in enumerate(coord_lines):
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"Bad coordinate line: {line}")
        
        mol['index'].append(i)
        mol['atom'].append(parts[0])
        mol['coordinates'].append([float(x) for x in parts[1:]])
    
    return mol


def distance_angstrom(coord1: List[float], coord2: List[float]) -> float:
    """Calculate Euclidean distance between two points in Angstroms."""
    return math.sqrt(sum((a - b)**2 for a, b in zip(coord1, coord2)))


class C4Baseline:
    """
    Computes C₄/R⁴ baseline for Zn²⁺-water induction energy.
    
    The baseline follows:
        E_ind,baseline = -∑ᵢ C₄ / Rᵢ⁴
    
    where the sum is over all Zn-O distances.
    """
    
    def __init__(self, 
                mol: Dict,
                C4: float = 1.0,
                kcal: bool = False,
                pairwise: bool = False):
        """
        Compute C4 baseline induction energy.
        
        :param mol: Dictionary with 'index', 'atom', 'coordinates'
        :param C4: C₄ coefficient (default 1.0, to be fitted later)
        :param kcal: If True, return energy in kcal/mol; else Hartree
        :param pairwise: If True, store per-water contributions
        """
        self.mol = mol
        self.C4 = C4
        self.kcal = kcal
        
        # Find Zn and O atoms
        self.zn_indices = [i for i, atom in enumerate(mol['atom']) if atom.lower() == 'zn']
        self.o_indices = [i for i, atom in enumerate(mol['atom']) if atom == 'O']
        
        if len(self.zn_indices) != 1:
            raise ValueError(f"Expected exactly 1 Zn atom, found {len(self.zn_indices)}")
        
        if len(self.o_indices) == 0:
            raise ValueError("No oxygen atoms found")
        
        self.zn_idx = self.zn_indices[0]
        self.zn_coord = mol['coordinates'][self.zn_idx]
        
        # Initialize energy accumulator
        self.e_c4 = 0.0
        
        # Pairwise contributions storage
        if pairwise:
            self.pairwise = {
                'i_zn': [],
                'j_o': [],
                'R_angstrom': [],
                'R_bohr': [],
                'inv_R4': [],
                'e_contribution': []
            }
        else:
            self.pairwise = None
        
        # Compute baseline
        self._compute_baseline(pairwise)
        
        # Total induction baseline
        self.e_ind = self.e_c4
    
    def _compute_baseline(self, pairwise: bool):
        """Internal method to compute the C4/R⁴ sum."""
        for o_idx in self.o_indices:
            o_coord = self.mol['coordinates'][o_idx]
            
            # Distance in Angstroms
            R_ang = distance_angstrom(self.zn_coord, o_coord)
            
            # Convert to Bohr for energy calculation (atomic units)
            R_bohr = R_ang * ANGSTROM_TO_BOHR
            
            # Avoid division by zero (shouldn't happen in valid geometries)
            if R_bohr < 1e-8:
                continue
            
            # Compute -C₄/R⁴ contribution (in Hartree)
            inv_R4 = 1.0 / (R_bohr ** 4)
            e_contrib = -self.C4 * inv_R4
            
            # Convert to kcal/mol if requested
            if self.kcal:
                e_contrib *= HARTREE_TO_KCAL
            
            # Accumulate
            self.e_c4 += e_contrib
            
            # Store pairwise if requested
            if pairwise:
                self.pairwise['i_zn'].append(self.zn_idx)
                self.pairwise['j_o'].append(o_idx)
                self.pairwise['R_angstrom'].append(R_ang)
                self.pairwise['R_bohr'].append(R_bohr)
                self.pairwise['inv_R4'].append(inv_R4)
                self.pairwise['e_contribution'].append(e_contrib)
    
    def get_energy(self) -> float:
        """Return total baseline induction energy."""
        return self.e_ind
    
    def get_zn_o_distances(self) -> List[float]:
        """Return list of Zn-O distances in Angstroms."""
        return [distance_angstrom(self.zn_coord, self.mol['coordinates'][o_idx]) 
                for o_idx in self.o_indices]
    
    def get_mean_inv_r4(self) -> float:
        """Return mean of 1/R⁴ values (useful for ML features)."""
        if not self.o_indices:
            return float('nan')
        
        inv_r4_values = []
        for o_idx in self.o_indices:
            o_coord = self.mol['coordinates'][o_idx]
            R_bohr = distance_angstrom(self.zn_coord, o_coord) * ANGSTROM_TO_BOHR
            if R_bohr > 1e-8:
                inv_r4_values.append(1.0 / (R_bohr ** 4))
        
        return sum(inv_r4_values) / len(inv_r4_values) if inv_r4_values else float('nan')


def main():
    parser = ArgumentParser(description="C4 baseline for Zn-water induction energy")
    parser.add_argument("xyz", help="XYZ file with Zn and water molecules")
    parser.add_argument("--C4", dest="C4", default=1.0, type=float, 
                    help="C₄ coefficient (default: 1.0, to be fitted)")
    parser.add_argument("--kcal", dest="kcal", action="store_true", default=False,
                    help="Print energy in kcal/mol (default: Hartree)")
    parser.add_argument("--pw", "--pairwise", dest="pairwise", action="store_true", 
                    default=False, help="Print pairwise contributions")
    
    args = parser.parse_args()
    
    # Read structure
    mol = read_xyz(args.xyz)
    
    # Compute baseline
    baseline = C4Baseline(mol, C4=args.C4, kcal=args.kcal, pairwise=args.pairwise)
    
    # Output
    name = args.xyz.replace('.xyz', '')
    
    if args.pairwise:
        print(f"C4 Baseline for {name}")
        print(f"C₄ coefficient: {args.C4}")
        print(f"\nPairwise contributions:")
        print(f"{'Zn_idx':<8} {'O_idx':<8} {'R(Å)':<10} {'R(bohr)':<12} {'1/R⁴':<15} {'E_contrib':<12}")
        print("-" * 75)
        
        pw = baseline.pairwise
        for i in range(len(pw['i_zn'])):
            print(f"{pw['i_zn'][i]:<8} {pw['j_o'][i]:<8} "
                f"{pw['R_angstrom'][i]:<10.4f} {pw['R_bohr'][i]:<12.4f} "
                f"{pw['inv_R4'][i]:<15.6e} {pw['e_contribution'][i]:<12.6f}")
        
        print("-" * 75)
        unit = "kcal/mol" if args.kcal else "Hartree"
        print(f"Total E_ind,baseline: {baseline.e_ind:.6f} {unit}")
    else:
        unit = "kcal/mol" if args.kcal else "Hartree"
        print(f"{name:<30} {baseline.e_ind:12.6f} {unit}")


if __name__ == "__main__":
    main()