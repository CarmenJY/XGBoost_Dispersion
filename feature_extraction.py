"""
Feature extraction for Zn2+ + (H2O)_n clusters (n=1..6).

Input: XYZ file containing 1 Zn atom and n water molecules.
Output: a small, physically motivated feature set for induction ML + C4 baseline.

Usage:
python feature_extraction.py path/to/cluster.xyz
python feature_extraction.py path/to/cluster.xyz --json
python feature_extraction.py path/to/cluster.xyz --csv
"""

from __future__ import annotations
import argparse
import json
import math
import statistics
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# Import C4 baseline
try:
    from baseline_c4 import C4Baseline
except ImportError:
    # If baseline_c4 is not in the same directory, provide a fallback
    C4Baseline = None


@dataclass
class Atom:
    sym: str
    x: float
    y: float
    z: float

    def pos(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


def dist(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.dist(a, b)


def read_xyz(path: str) -> List[Atom]:
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    if len(lines) < 3:
        raise ValueError("XYZ file too short.")

    try:
        nat = int(lines[0])
    except ValueError as e:
        raise ValueError("First line must be number of atoms.") from e

    coord_lines = lines[2:2 + nat]
    if len(coord_lines) != nat:
        raise ValueError(f"Expected {nat} coordinate lines, got {len(coord_lines)}.")

    atoms: List[Atom] = []
    for ln in coord_lines:
        parts = ln.split()
        if len(parts) != 4:
            raise ValueError(f"Bad coordinate line: {ln}")
        sym = parts[0]
        x, y, z = map(float, parts[1:])
        atoms.append(Atom(sym, x, y, z))

    return atoms


def atoms_to_mol_dict(atoms: List[Atom]) -> Dict:
    """Convert Atom list to baseline_c4 mol dict format."""
    mol = {
        'index': list(range(len(atoms))),
        'atom': [a.sym for a in atoms],
        'coordinates': [[a.x, a.y, a.z] for a in atoms]
    }
    return mol


def unit(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    norm = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    if norm == 0:
        return (0.0, 0.0, 0.0)
    return (v[0]/norm, v[1]/norm, v[2]/norm)


def angle_deg(a: Tuple[float, float, float],
            b: Tuple[float, float, float],
            c: Tuple[float, float, float]) -> float:
    """
    Angle ABC in degrees (with B as vertex).
    """
    ba = (a[0]-b[0], a[1]-b[1], a[2]-b[2])
    bc = (c[0]-b[0], c[1]-b[1], c[2]-b[2])
    uba = unit(ba)
    ubc = unit(bc)
    dot = max(-1.0, min(1.0, uba[0]*ubc[0] + uba[1]*ubc[1] + uba[2]*ubc[2]))
    return math.degrees(math.acos(dot))


def assign_waters(atoms: List[Atom],
                oh_max: float = 1.25) -> List[Tuple[int, int, int]]:
    """
    Assign waters by pairing each O with its two nearest H atoms within a cutoff.
    Returns list of (O_index, H1_index, H2_index).
    """
    O_indices = [i for i, a in enumerate(atoms) if a.sym == "O"]
    H_indices = [i for i, a in enumerate(atoms) if a.sym == "H"]

    waters: List[Tuple[int, int, int]] = []

    for oi in O_indices:
        o_pos = atoms[oi].pos()
        # compute distances to all H
        dists = []
        for hi in H_indices:
            d = dist(o_pos, atoms[hi].pos())
            dists.append((d, hi))
        dists.sort(key=lambda x: x[0])

        # pick two closest H within cutoff
        close = [hi for (d, hi) in dists if d <= oh_max]
        if len(close) < 2:
            # Not enough H near this oxygen; skip (or raise)
            continue
        h1, h2 = close[0], close[1]
        waters.append((oi, h1, h2))

    # De-duplicate if the same H got assigned twice (rare in clean cluster data)
    # We'll keep it simple: ensure each O has a unique tuple, and prefer local assignments.
    unique = []
    used_H = set()
    for (oi, h1, h2) in sorted(waters, key=lambda t: t[0]):
        if h1 in used_H or h2 in used_H:
            continue
        used_H.add(h1); used_H.add(h2)
        unique.append((oi, h1, h2))

    return unique


def summarize(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"min": float("nan"), "mean": float("nan"), "max": float("nan"), "std": float("nan")}
    if len(vals) == 1:
        return {"min": vals[0], "mean": vals[0], "max": vals[0], "std": 0.0}
    return {
        "min": min(vals),
        "mean": statistics.fmean(vals),
        "max": max(vals),
        "std": statistics.pstdev(vals),
    }


def count_hbonds(atoms: List[Atom],
                waters: List[Tuple[int, int, int]],
                hbond_HO_max: float = 2.5,
                angle_min_deg: float = 150.0) -> int:
    """
    Simple water-water H-bond count using:
    H···O distance <= 2.5 Å and angle O-H···O >= 150°
    """
    hb = 0
    # For each donor water (O_d, H1, H2), each acceptor water O_a
    for i, (od, h1, h2) in enumerate(waters):
        Od = atoms[od].pos()
        for j, (oa, _, _) in enumerate(waters):
            if i == j:
                continue
            Oa = atoms[oa].pos()
            for hi in (h1, h2):
                H = atoms[hi].pos()
                dHO = dist(H, Oa)
                if dHO <= hbond_HO_max:
                    ang = angle_deg(Od, H, Oa)  # O_d - H - O_a
                    if ang >= angle_min_deg:
                        hb += 1
    return hb


def extract_features(xyz_path: str, compute_baseline: bool = True) -> Dict[str, float]:
    """
    Extract features and optionally compute C4 baseline.
    
    Args:
        xyz_path: Path to XYZ file
        compute_baseline: If True, compute and include E_baseline_C4
    
    Returns:
        Dictionary of features (and baseline if requested)
    """
    atoms = read_xyz(xyz_path)

    zn_indices = [i for i, a in enumerate(atoms) if a.sym.lower() == "zn"]
    if len(zn_indices) != 1:
        raise ValueError(f"Expected exactly 1 Zn atom, found {len(zn_indices)}.")
    zn_i = zn_indices[0]
    Zn = atoms[zn_i].pos()

    waters = assign_waters(atoms)
    n_waters = len(waters)
    if n_waters == 0:
        raise ValueError("No water molecules detected (could not assign O-H bonds).")

    O_indices = [oi for (oi, _, _) in waters]
    O_positions = [atoms[oi].pos() for oi in O_indices]

    # Zn–O distances
    zn_o = [dist(Zn, Opos) for Opos in O_positions]
    zn_o_stats = summarize(zn_o)

    # Inverse-distance moments (help ML learn steep distance scaling)
    # Avoid division by zero (should not happen in valid geometries)
    inv2 = [1.0/(r*r) for r in zn_o if r > 1e-8]
    inv3 = [1.0/(r*r*r) for r in zn_o if r > 1e-8]
    inv4 = [1.0/(r*r*r*r) for r in zn_o if r > 1e-8]

    # Coordination counts (within cutoffs)
    def count_within(cut: float) -> int:
        return sum(1 for r in zn_o if r <= cut)

    coord_23 = count_within(2.3)
    coord_25 = count_within(2.5)
    coord_30 = count_within(3.0)

    # Water–water O–O geometry
    oo = []
    for i in range(len(O_positions)):
        for j in range(i+1, len(O_positions)):
            oo.append(dist(O_positions[i], O_positions[j]))
    oo_stats = summarize(oo) if oo else {"min": float("nan"), "mean": float("nan"), "max": float("nan"), "std": float("nan")}

    # Hydrogen bond count among waters (optional but useful for n>=3)
    hb_count = count_hbonds(atoms, waters) if n_waters >= 2 else 0

    feats: Dict[str, float] = {
        # Size / identity
        "n_waters": float(n_waters),

        # Zn–O distance summaries (Å)
        "ZnO_min": zn_o_stats["min"],
        "ZnO_mean": zn_o_stats["mean"],
        "ZnO_max": zn_o_stats["max"],
        "ZnO_std": zn_o_stats["std"],

        # inverse-distance moments
        "mean_invR2": statistics.fmean(inv2) if inv2 else float("nan"),
        "mean_invR3": statistics.fmean(inv3) if inv3 else float("nan"),
        "mean_invR4": statistics.fmean(inv4) if inv4 else float("nan"),

        # coordination counts
        "coord_ZnO_le_2p3": float(coord_23),
        "coord_ZnO_le_2p5": float(coord_25),
        "coord_ZnO_le_3p0": float(coord_30),

        # water-water structure
        "OO_mean": oo_stats["mean"],
        "OO_min": oo_stats["min"],
        "OO_std": oo_stats["std"],

        # simple H-bond count proxy
        "HB_count": float(hb_count),
    }

    # Compute C4 baseline if requested and available
    if compute_baseline and C4Baseline is not None:
        try:
            mol_dict = atoms_to_mol_dict(atoms)
            baseline = C4Baseline(mol_dict, C4=1.0, kcal=True, pairwise=False)
            feats["E_baseline_C4"] = baseline.get_energy()
        except Exception as e:
            # If baseline computation fails, set to NaN and continue
            feats["E_baseline_C4"] = float("nan")
            import sys
            print(f"Warning: Could not compute baseline for {xyz_path}: {e}", file=sys.stderr)
    elif compute_baseline and C4Baseline is None:
        feats["E_baseline_C4"] = float("nan")
        import sys
        print("Warning: baseline_c4 module not found, E_baseline_C4 set to NaN", file=sys.stderr)

    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz", help="XYZ file with Zn and (H2O)_n cluster")
    ap.add_argument("--json", action="store_true", help="Print as JSON")
    ap.add_argument("--csv", action="store_true", help="Print as one-line CSV (header + value)")
    ap.add_argument("--no-baseline", action="store_true", help="Skip baseline computation")
    args = ap.parse_args()

    compute_baseline = not args.no_baseline
    feats = extract_features(args.xyz, compute_baseline=compute_baseline)

    if args.json:
        print(json.dumps(feats, indent=2, sort_keys=True))
        return

    if args.csv:
        keys = list(feats.keys())
        print(",".join(keys))
        print(",".join(str(feats[k]) for k in keys))
        return

    # default: pretty print
    for k in sorted(feats.keys()):
        print(f"{k:18s}  {feats[k]}")


if __name__ == "__main__":
    main()