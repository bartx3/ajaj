import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from ucimlrepo import fetch_ucirepo

from anonimize_data import MondrianAnonymizer

import pandas as pd

# wykresy
import matplotlib.pyplot as plt
from collections import Counter


def _run_anonymizer(df: pd.DataFrame, l: float, t: float):
    anonymizer = MondrianAnonymizer(
        data=df,
        k=5,
        l=l,
        tau=0.2,
        t=t,
        qi_continuous=['age'],
        qi_categorical=['sex', 'race'],
        sensitive_col='occupation'
    )
    _, partitions = anonymizer.run()
    return anonymizer, partitions


def _run_anonymizer_batch(df: pd.DataFrame, pairs: list[tuple[float, float]]):
    results: dict[tuple[float, float], tuple[MondrianAnonymizer, list]] = {}
    max_workers = min(8, len(pairs)) or 1

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_anonymizer, df, l_val, t_val): (l_val, t_val)
            for l_val, t_val in pairs
        }
        for future in as_completed(futures):
            pair = futures[future]
            results[pair] = future.result()

    return results


def _group_size_distribution(partitions) -> dict[int, int]:
    group_sizes = [len(group) for group in partitions]
    return dict(Counter(group_sizes))


def _new_histogram_figure(title: str):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlabel("Rozmiar grupy EC")
    ax.set_ylabel("Liczba grup o danym rozmiarze")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    return fig, ax


def add_histogram(ax, partitions, k: float, l: float, t: float):
    """
    Dodaje pojedyncza linie histogramu grup dla konkretnej konfiguracji k, l, t.
    Funkcja moze byc wywolywana wielokrotnie, aby narysowac kilka linii na jednym wykresie.
    """
    dist = _group_size_distribution(partitions)
    sizes = sorted(dist.keys())
    counts = [dist[s] for s in sizes]
    ax.plot(sizes, counts, marker='o', label=f"k={k}, l={l}, t={t}")


def _finalize_and_save_histogram(fig, ax, out_file: str):
    x_ticks = sorted(
        {
            int(tick)
            for line in ax.lines
            for tick in line.get_xdata()
        }
    )
    if x_ticks:
        ax.set_xticks(x_ticks)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


def _plot_distribution_by_param(
    distributions: dict[float, dict[int, int]],
    out_file: str,
    title: str,
    x_label: str
):
    all_sizes = sorted({size for dist in distributions.values() for size in dist.keys()})

    plt.figure(figsize=(14, 7))
    for param, dist in distributions.items():
        y_values = [dist.get(size, 0) for size in all_sizes]
        plt.plot(all_sizes, y_values, marker='o', label=f"{x_label}={param}")

    plt.xlabel("Rozmiar grupy EC")
    plt.ylabel("Liczba grup o danym rozmiarze")
    plt.title(title)
    plt.xticks(all_sizes)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def _plot_group_count(
    values: list[float],
    counts: list[int],
    out_file: str,
    title: str,
    x_label: str
):
    plt.figure(figsize=(10, 6))
    plt.plot(values, counts, marker='o', linewidth=2)
    plt.xlabel(x_label)
    plt.ylabel("Liczba grup")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def _plot_group_count_series(
    values: list[float],
    series_by_label: dict[str, list[int]],
    out_file: str,
    title: str,
    x_label: str
):
    plt.figure(figsize=(10, 6))
    for label, counts in series_by_label.items():
        plt.plot(values, counts, marker='o', linewidth=2, label=label)
    plt.xlabel(x_label)
    plt.ylabel("Liczba grup")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()

class MyTestCase(unittest.TestCase):
    def test_mondrian_anonimizer(self):
        
        # 1. Import dataset
        adult = fetch_ucirepo(id=2)

        X = adult.data.features
        y = adult.data.targets

        df = pd.concat([X, y], axis=1).dropna()
        df.columns = df.columns.str.strip()
        df['income'] = df['income'].str.strip()

        l_values = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
        t_values = [1.4, 1.5, 1.6]
        k_value = 5
        fixed_t_for_l_scan = 1.5
        selected_l_for_t_scans = [1.5, 1.8, 2.1, 2.4, 2.8]

        main_pairs = [(l_val, t_val) for l_val in l_values for t_val in t_values]
        main_runs = _run_anonymizer_batch(df, main_pairs)

        selected_pairs = [
            (l_fixed, t_val)
            for l_fixed in selected_l_for_t_scans
            for t_val in t_values
        ]
        selected_runs = _run_anonymizer_batch(df, selected_pairs)

        # 1) Wykres liczby grup dla roznych l z trzema liniami dla t
        group_counts_by_t: dict[str, list[int]] = {f"t={t_val}": [] for t_val in t_values}

        for t_val in t_values:
            for l_val in l_values:
                anonymizer, partitions = main_runs[(l_val, t_val)]
                group_counts_by_t[f"t={t_val}"].append(len(partitions))

                print(f"l={l_val}, t={t_val}, liczba_grup={len(partitions)}")
                assert all(anonymizer.check_k_anonimity(partition) for partition in partitions)
                assert all(anonymizer.check_l_divergence_ENTROPY(partition) for partition in partitions)
                assert all(anonymizer.check_t_closeness_TVD(partition) for partition in partitions)

        _plot_group_count_series(
            values=l_values,
            series_by_label=group_counts_by_t,
            out_file="liczba_grup_wg_l.png",
            title="Liczba grup dla roznych l i t",
            x_label="l"
        )

        # 2) i 3) Dystrybucja mocy grup dla roznych t przy dwoch wybranych l
        # 2) i 3) Dystrybucja mocy grup dla roznych t - zestawienie wybranych l na jednym wykresie
        fig_t_comb, ax_t_comb = _new_histogram_figure(
            f"Dystrybucja mocy grup dla roznych t (k={k_value}, l={selected_l_for_t_scans})"
        )

        for l_fixed in selected_l_for_t_scans:
            group_counts_by_t: list[int] = []

            for t_val in t_values:
                anonymizer, partitions = selected_runs[(l_fixed, t_val)]
                add_histogram(ax_t_comb, partitions, k=k_value, l=l_fixed, t=t_val)
                group_counts_by_t.append(len(partitions))

                print(f"l={l_fixed}, t={t_val}, liczba_grup={len(partitions)}")
            assert all(anonymizer.check_k_anonimity(partition) for partition in partitions)
            assert all(anonymizer.check_l_divergence_ENTROPY(partition) for partition in partitions)
            assert all(anonymizer.check_t_closeness_TVD(partition) for partition in partitions)

            # zachowaj tez wykres liczby grup vs t dla kazdego l osobno
            l_tag = str(l_fixed).replace('.', '_')
            _plot_group_count(
            values=t_values,
            counts=group_counts_by_t,
            out_file=f"liczba_grup_wg_t_l_{l_tag}.png",
            title=f"Liczba grup dla roznych t (l={l_fixed})",
            x_label="t"
            )

        _finalize_and_save_histogram(
            fig_t_comb,
            ax_t_comb,
            "dystrybucja_mocy_grup_wg_t_zestawione.png"
        )

        # 4) Drugi histogram rozkladu liczności grup dla wszystkich l i t
        fig_all, ax_all = _new_histogram_figure(
            f"Dystrybucja liczności grup dla wszystkich l i t (k={k_value})"
        )

        for l_val in l_values:
            for t_val in t_values:
                _, partitions = main_runs[(l_val, t_val)]
                add_histogram(ax_all, partitions, k=k_value, l=l_val, t=t_val)

        _finalize_and_save_histogram(
            fig_all,
            ax_all,
            "dystrybucja_licznosci_grup_wszystkie_l_i_t.png"
        )


if __name__ == '__main__':
    unittest.main()
