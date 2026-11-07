"""Reproduce focused pipeline and dashboard checks: make check."""
import hashlib
import json
from datetime import date
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

import charts
from data_loading import load_records, records_to_frame
from stats import bootstrap_difference, permutation_proportions, run_hypotheses

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/raw/nizhegorodskaia-oblast.geojson'


def check_pipeline():
    frame = records_to_frame(load_records(SOURCE))
    metadata = json.loads(SOURCE.with_suffix('.source.json').read_text())
    results = json.loads((ROOT / 'solution/results.json').read_text())
    with SOURCE.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert digest == metadata['sha256'] == results['source_sha256']
    assert len(frame) == metadata['records'] == results['data']['rows']
    assert frame.source_row.tolist() == list(range(len(frame)))
    assert int(frame.id.duplicated().sum()) == metadata['duplicate_nonempty_ids']
    assert (frame.injured_count + frame.dead_count <= frame.participants_count).all()
    assert frame.fatal.equals(frame.dead_count.gt(0))
    assert frame.map_valid.equals(frame.global_valid & frame.inside_region_box)
    swapped = frame.coordinates_swapped
    np.testing.assert_array_equal(frame.loc[swapped, 'longitude'], frame.loc[swapped, 'latitude_raw'])
    np.testing.assert_array_equal(frame.loc[swapped, 'latitude'], frame.loc[swapped, 'longitude_raw'])
    qa = results['data']['map']
    assert qa['valid'] == int(frame.map_valid.sum())
    assert qa['excluded'] == qa['missing'] + qa['zeros'] + qa['other']
    assert qa['valid'] + qa['excluded'] == len(frame)
    return frame, results


def check_statistics(frame, results):
    # For four binary events, enumerate all six relabelings independently.
    differences = []
    for group in combinations(range(4), 2):
        positives = sum(index < 2 for index in group)
        differences.append(positives / 2 - (2 - positives) / 2)
    exact_p = np.mean(np.abs(differences) >= 1)
    p, extreme = permutation_proportions([1, 1], [0, 0])
    assert abs(p - exact_p) < .02
    assert p == (extreme + 1) / 10000
    assert (p, extreme) == permutation_proportions([1, 1], [0, 0])
    assert bootstrap_difference([2, 2], [1, 1]) == [1., 1.]
    full = frame[frame.year.between(2015, 2025)]
    checked = run_hypotheses(frame)
    assert checked == run_hypotheses(full) == results['hypotheses']
    first, second, third = checked['results']
    assert first['n_a'] + first['n_b'] == len(full)
    assert second['n_a'] + second['n_b'] == int((full.night & full.category.eq('Столкновение')).sum())
    assert third['n_a'] + third['n_b'] == int(full.season.isin(['зима', 'лето']).sum())
    for item in checked['results']:
        assert 0 < item['p_value'] <= 1 and item['ci_low'] <= item['ci_high']
        assert np.isfinite([item['effect'], item['ci_low'], item['ci_high']]).all()


def check_charts_and_projections(frame, results):
    monthly = charts.monthly(frame, frame.datetime.min(), frame.datetime.max())
    assert sum(monthly.data[0].y) == len(frame)
    assert sum(charts.categories(frame).data[0].x) == len(frame)
    assert charts.injured_ecdf(frame).data[0].y[-1] == 100
    assert int(np.asarray(charts.hour_weekday(frame).data[0].customdata)[:, :, 0].sum()) == len(frame)
    grid = charts.grid_map(frame, backdrop=False)
    assert sum(grid.data[0].z) == int(frame.map_valid.sum())
    points = charts.point_map(frame, backdrop=False)
    assert sum(len(trace.lat) for trace in points.data) == min(5000, int(frame.map_valid.sum()))
    empty = frame.iloc[:0]
    assert charts.monthly(empty, date(2020, 1, 1), date(2020, 1, 31)) is None
    assert charts.point_map(empty) is None and charts.grid_map(empty) is None
    projected = pd.read_csv(ROOT / 'solution/cache/projections.csv')
    metadata = results['projections']['metadata']
    assert len(projected) == metadata['sample_rows'] and projected.source_row.is_unique
    assert projected.year.between(2015, 2025).all()
    assert set(projected.source_row) <= set(frame.source_row)
    assert np.isfinite(projected[['pca1', 'pca2', 'tsne15_1', 'tsne15_2', 'tsne40_1', 'tsne40_2']]).all().all()
    assert not {'severity', 'fatal', 'injured_count', 'dead_count'} & set(metadata['input_features'])
    assert set(metadata['tsne']) == {'15', '40'}


def check_app(frame):
    at = AppTest.from_file(str(ROOT / 'solution/app.py'), default_timeout=120).run()

    def assert_metrics(expected):
        assert not at.exception, [item.message for item in at.exception]
        values = [len(expected), int(expected.injured_count.sum()),
                  int(expected.fatal.sum()), int((~expected.map_valid).sum())]
        assert [int(item.value.replace(',', '')) for item in at.metric] == values

    assert_metrics(frame)
    assert len(at.get('plotly_chart')) == 12
    at.checkbox(key='filter_all_category').uncheck().run()
    assert not at.exception and len(at.metric) == 0
    assert any('нет ДТП' in item.value for item in at.info)
    at.multiselect(key='filter_category').set_value(['Столкновение']).run()
    selected = frame[frame.category.eq('Столкновение')]
    assert_metrics(selected)
    at.checkbox(key='filter_all_region').uncheck().run()
    region = selected.region.value_counts().index[0]
    at.multiselect(key='filter_region').set_value([region]).run()
    selected = selected[selected.region.eq(region)]
    assert_metrics(selected)
    start, end = date(2019, 1, 1), date(2019, 1, 31)
    at.date_input(key='filter_period').set_value((start, end)).run()
    expected = selected[selected.datetime.between(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1), inclusive='left')]
    assert_metrics(expected)
    at.date_input(key='filter_period').set_value((start,)).run()
    assert not at.exception and len(at.metric) == 0
    assert any('обе границы' in item.value for item in at.info)
    at.button[0].click().run()
    assert_metrics(frame)
    print('AppTest: default, empty selection, category, region, dates and reset passed.')


if __name__ == '__main__':
    started = perf_counter()
    if not SOURCE.exists() or not (ROOT / 'solution/cache/projections.csv').exists():
        raise SystemExit('Сначала выполните python data/download_data.py и python solution/analyze.py.')
    frame, results = check_pipeline()
    check_statistics(frame, results)
    check_charts_and_projections(frame, results)
    check_app(frame)
    print(f'All checks passed in {perf_counter() - started:.1f}s. {len(frame):,} source events preserved.')
