#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import uuid
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class GroupStats:
    gid: int
    size: int
    entropy: float
    t_distance: float


def parse_csv_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def add_anonymous_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["anon_id"] = [str(uuid.uuid4())[:8] for _ in range(len(df))]
    return df


def numeric_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    out = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


def build_distance_frame(df: pd.DataFrame, qi_cols: list[str]) -> pd.DataFrame:
    work = pd.DataFrame(index=df.index)
    for c in qi_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            col = df[c].astype(float)
            std = col.std(ddof=0)
            work[c] = 0.0 if std == 0 else (col - col.mean()) / std
        else:
            # Proste kodowanie kategorii do dystansu euklidesowego.
            work[c] = pd.Categorical(df[c].astype(str)).codes.astype(float)
    return work


def greedy_k_partition(df: pd.DataFrame, qi_cols: list[str], k: int) -> pd.Series:
    feat = build_distance_frame(df, qi_cols).to_numpy()
    idx = np.arange(len(df))
    unassigned = set(idx.tolist())
    groups: list[list[int]] = []

    while len(unassigned) >= k:
        seed = min(unassigned)
        unassigned.remove(seed)
        if not unassigned:
            groups.append([seed])
            break
        cand = np.array(sorted(unassigned))
        d = np.linalg.norm(feat[cand] - feat[seed], axis=1)
        take = cand[np.argsort(d)[: max(0, k - 1)]].tolist()
        for t in take:
            unassigned.remove(t)
        groups.append([seed] + take)

    leftovers = sorted(unassigned)
    if leftovers:
        if not groups:
            groups.append(leftovers)
        else:
            centers = [feat[g].mean(axis=0) for g in groups]
            for row_idx in leftovers:
                d = [float(np.linalg.norm(feat[row_idx] - c)) for c in centers]
                best_g = int(np.argmin(d))
                groups[best_g].append(row_idx)
                centers[best_g] = feat[groups[best_g]].mean(axis=0)

    gid = np.empty(len(df), dtype=int)
    for i, g in enumerate(groups):
        for row_idx in g:
            gid[row_idx] = i
    return pd.Series(gid, index=df.index, name="group_id")


def entropy_of_series(s: pd.Series) -> float:
    p = s.value_counts(normalize=True)
    if p.empty:
        return 0.0
    return float(-(p * np.log(p)).sum())


def distribution_distance_l1(p: pd.Series, q: pd.Series) -> float:
    all_idx = p.index.union(q.index)
    p2 = p.reindex(all_idx, fill_value=0.0)
    q2 = q.reindex(all_idx, fill_value=0.0)
    return float(np.abs(p2 - q2).sum())


def group_centroid_distance(
    df: pd.DataFrame, feat: np.ndarray, gid_a: int, gid_b: int
) -> float:
    a_rows = np.where(df["group_id"].to_numpy() == gid_a)[0]
    b_rows = np.where(df["group_id"].to_numpy() == gid_b)[0]
    if len(a_rows) == 0 or len(b_rows) == 0:
        return float("inf")
    ca = feat[a_rows].mean(axis=0)
    cb = feat[b_rows].mean(axis=0)
    return float(np.linalg.norm(ca - cb))


def merge_group_into(df: pd.DataFrame, source_gid: int, target_gid: int) -> None:
    df.loc[df["group_id"] == source_gid, "group_id"] = target_gid


def compact_group_ids(df: pd.DataFrame) -> None:
    mapping = {old: new for new, old in enumerate(sorted(df["group_id"].unique()))}
    df["group_id"] = df["group_id"].map(mapping).astype(int)


def enforce_l_diversity(
    df: pd.DataFrame, qi_cols: list[str], sensitive_col: str, l: int
) -> None:
    feat = build_distance_frame(df, qi_cols).to_numpy()
    target_entropy = math.log(l)

    changed = True
    while changed:
        changed = False
        bad_groups = []
        for gid, g in df.groupby("group_id"):
            if entropy_of_series(g[sensitive_col]) < target_entropy:
                bad_groups.append(int(gid))
        if not bad_groups:
            break

        for bad_gid in bad_groups:
            all_gids = sorted(df["group_id"].unique())
            candidates = [x for x in all_gids if x != bad_gid]
            if not candidates:
                continue
            dists = [group_centroid_distance(df, feat, bad_gid, c) for c in candidates]
            nearest = candidates[int(np.argmin(dists))]
            merge_group_into(df, bad_gid, nearest)
            changed = True
        if changed:
            compact_group_ids(df)


def enforce_t_closeness(
    df: pd.DataFrame, qi_cols: list[str], sensitive_col: str, t: float
) -> None:
    feat = build_distance_frame(df, qi_cols).to_numpy()
    global_dist = df[sensitive_col].value_counts(normalize=True)

    changed = True
    while changed:
        changed = False
        over_t: list[int] = []
        for gid, g in df.groupby("group_id"):
            local = g[sensitive_col].value_counts(normalize=True)
            dist = distribution_distance_l1(local, global_dist)
            if dist > t:
                over_t.append(int(gid))
        if not over_t:
            break

        for bad_gid in over_t:
            all_gids = sorted(df["group_id"].unique())
            candidates = [x for x in all_gids if x != bad_gid]
            if not candidates:
                continue
            dists = [group_centroid_distance(df, feat, bad_gid, c) for c in candidates]
            nearest = candidates[int(np.argmin(dists))]
            merge_group_into(df, bad_gid, nearest)
            changed = True
        if changed:
            compact_group_ids(df)


def generalize_group_values(df: pd.DataFrame, qi_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for gid, g in out.groupby("group_id"):
        idx = g.index
        for c in qi_cols:
            if pd.api.types.is_numeric_dtype(out[c]):
                lo = float(g[c].min())
                hi = float(g[c].max())
                out.loc[idx, c] = f"[{lo:.2f}, {hi:.2f}]"
            else:
                vals = sorted(g[c].astype(str).unique().tolist())
                out.loc[idx, c] = "{" + ",".join(vals) + "}"
    return out


def gather_group_stats(df: pd.DataFrame, sensitive_col: str) -> list[GroupStats]:
    global_dist = df[sensitive_col].value_counts(normalize=True)
    stats: list[GroupStats] = []
    for gid, g in df.groupby("group_id"):
        local = g[sensitive_col].value_counts(normalize=True)
        stats.append(
            GroupStats(
                gid=int(gid),
                size=int(len(g)),
                entropy=entropy_of_series(g[sensitive_col]),
                t_distance=distribution_distance_l1(local, global_dist),
            )
        )
    return sorted(stats, key=lambda x: x.gid)


def print_attack_report(
    df: pd.DataFrame, sensitive_col: str, victim_index: int, tau: float
) -> None:
    if victim_index < 0 or victim_index >= len(df):
        victim_index = 0
    victim_gid = int(df.iloc[victim_index]["group_id"])
    global_dist = df[sensitive_col].value_counts(normalize=True)
    local_dist = (
        df.loc[df["group_id"] == victim_gid, sensitive_col].value_counts(normalize=True)
    )

    all_vals = sorted(global_dist.index.union(local_dist.index).tolist())
    print("\n=== Atak Skośności ===")
    print(f"Ofiara: wiersz={victim_index}, grupa={victim_gid}")
    print("wartosc\tP_global\tP_local\t|delta|")
    success = False
    for v in all_vals:
        pg = float(global_dist.get(v, 0.0))
        pl = float(local_dist.get(v, 0.0))
        delta = abs(pl - pg)
        if delta > tau:
            success = True
        print(f"{v}\t{pg:.4f}\t{pl:.4f}\t{delta:.4f}")
    print(f"Wynik ataku (tau={tau}): {'UDANY' if success else 'NIEUDANY'}")


def run_pipeline(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.csv)
    qi_cols = parse_csv_list(args.qi)
    direct_id_cols = parse_csv_list(args.direct_identifiers)
    sensitive_col = args.sensitive

    missing = [c for c in qi_cols + [sensitive_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Brak kolumn w CSV: {missing}")

    drop_cols = [c for c in direct_id_cols if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    df = add_anonymous_id(df)

    if len(df) < args.k:
        raise ValueError("Liczba rekordow jest mniejsza niz k.")

    df["group_id"] = greedy_k_partition(df, qi_cols, args.k)
    enforce_l_diversity(df, qi_cols, sensitive_col, args.l)

    print("\n=== Statystyki po k-anon + l-diversity ===")
    for st in gather_group_stats(df, sensitive_col):
        print(
            f"grupa={st.gid:3d} rozmiar={st.size:3d} "
            f"entropia={st.entropy:.4f} dist_global={st.t_distance:.4f}"
        )

    print_attack_report(df, sensitive_col, args.victim_index, args.tau)

    enforce_t_closeness(df, qi_cols, sensitive_col, args.t)

    print("\n=== Statystyki po uszczelnieniu t-closeness ===")
    for st in gather_group_stats(df, sensitive_col):
        print(
            f"grupa={st.gid:3d} rozmiar={st.size:3d} "
            f"entropia={st.entropy:.4f} dist_global={st.t_distance:.4f}"
        )

    print_attack_report(df, sensitive_col, args.victim_index, args.tau)

    anon = generalize_group_values(df, qi_cols)
    return anon


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Greedy privacy pipeline: k-anonimowosc, l-roznorodnosc, atak skosnosci, "
            "t-bliskosc"
        )
    )
    p.add_argument("--csv", required=True, help="Sciezka do pliku CSV")
    p.add_argument(
        "--qi", required=True, help="Kolumny quasi-identyfikatorow, np. age,sex,zip"
    )
    p.add_argument(
        "--sensitive", required=True, help="Kolumna atrybutu wrazliwego (S)"
    )
    p.add_argument(
        "--direct-identifiers",
        default="",
        help="Kolumny bezposrednich identyfikatorow do usuniecia, np. name,ssn",
    )
    p.add_argument("--k", type=int, default=5, help="Parametr k-anonimowosci")
    p.add_argument("--l", type=int, default=2, help="Parametr l-roznorodnosci")
    p.add_argument("--t", type=float, default=0.5, help="Prog t-bliskosci (L1)")
    p.add_argument(
        "--tau", type=float, default=0.3, help="Prog sukcesu ataku skosnosci"
    )
    p.add_argument(
        "--victim-index", type=int, default=0, help="Indeks ofiary do symulacji ataku"
    )
    p.add_argument(
        "--out", default="anonymized_output.csv", help="Plik wyjsciowy zanonimizowany"
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    anon = run_pipeline(args)
    anon.to_csv(args.out, index=False)
    print(f"\nZapisano zanonimizowany zbior do: {args.out}")


if __name__ == "__main__":
    main()
