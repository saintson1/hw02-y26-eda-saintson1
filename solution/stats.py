"""Три проверки на ДТП за полные 2015-2025 годы."""
import numpy as np
from scipy import stats

SEED = 42
BOOTSTRAP_REPEATS = 5000
PERMUTATIONS = 9999


def bootstrap_difference(first, second, seed=SEED):
    rng = np.random.default_rng(seed)
    means = []
    for sample in (first, second):
        values, counts = np.unique(np.asarray(sample, dtype=float), return_counts=True)
        # Выбор частот эквивалентен bootstrap строк для среднего.
        repeats = rng.multinomial(len(sample), counts / len(sample), size=BOOTSTRAP_REPEATS)
        means.append(repeats @ values / len(sample))
    return np.quantile(means[0] - means[1], [.025, .975]).tolist()


def permutation_proportions(first, second, seed=SEED):
    first, second = np.asarray(first, dtype=int), np.asarray(second, dtype=int)
    assert np.isin(first, [0, 1]).all() and np.isin(second, [0, 1]).all()
    observed = first.mean() - second.mean()
    positives = int(first.sum() + second.sum())
    rng = np.random.default_rng(seed)
    # Число единиц первой группы при случайном перераспределении бинарных исходов.
    hits = rng.hypergeometric(positives, len(first) + len(second) - positives, len(first), size=PERMUTATIONS)
    differences = hits / len(first) - (positives - hits) / len(second)
    extreme = int(np.sum(abs(differences) >= abs(observed) - 1e-12))
    return (extreme + 1) / (PERMUTATIONS + 1), extreme


def run_hypotheses(frame):
    full = frame[frame.year.between(2015, 2025)]
    collisions = full[full.night & full.category.eq('Столкновение')]
    groups = [
        ('H1', 'Доля ночных ДТП: выходные против будней', 'Выходные', 'Будни',
         full.loc[full.weekend, 'night'], full.loc[~full.weekend, 'night'], 'permutation'),
        ('H2', 'Раненые в ночных столкновениях: выходные против будней', 'Выходные', 'Будни',
         collisions.loc[collisions.weekend, 'injured_count'],
         collisions.loc[~collisions.weekend, 'injured_count'], 'Welch'),
        ('H3', 'Доля наездов на пешеходов: зима против лета', 'Зима', 'Лето',
         full.loc[full.season.eq('зима'), 'category'].eq('Наезд на пешехода'),
         full.loc[full.season.eq('лето'), 'category'].eq('Наезд на пешехода'), 'permutation'),
    ]
    results = []
    for i, (identifier, question, group_a, group_b, first, second, test) in enumerate(groups):
        first, second = first.dropna().to_numpy(dtype=float), second.dropna().to_numpy(dtype=float)
        if min(len(first), len(second)) < 2:
            raise ValueError(f'{identifier}: недостаточно наблюдений')
        seed = SEED + i
        if test == 'Welch':
            p_value, extreme = float(stats.ttest_ind(first, second, equal_var=False).pvalue), None
        else:
            p_value, extreme = permutation_proportions(first, second, seed)
        scale = 1 if test == 'Welch' else 100
        low, high = np.asarray(bootstrap_difference(first, second, seed)) * scale
        results.append(dict(id=identifier, question=question, group_a=group_a, group_b=group_b,
                            n_a=len(first), n_b=len(second), mean_a=float(first.mean()), mean_b=float(second.mean()),
                            test=test, effect=float((first.mean() - second.mean()) * scale),
                            unit='раненых на ДТП' if test == 'Welch' else 'п.п.',
                            ci_low=float(low), ci_high=float(high), p_value=p_value,
                            reject=p_value < .05, seed=seed, extreme_permutations=extreme))
    return dict(period='2015-2025', full_year_rows=len(full), alpha=.05, ci_level=.95,
                bootstrap_repeats=BOOTSTRAP_REPEATS, permutations=PERMUTATIONS, results=results)
