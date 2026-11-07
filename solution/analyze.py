"""Пересчитать результаты и графики: make analyze."""
import hashlib
import json
from pathlib import Path

import numpy as np

import charts
from data_loading import load_records, records_to_frame
from projections import compute_projections
from stats import run_hypotheses

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/raw/nizhegorodskaia-oblast.geojson'


def data_summary(frame):
    counts = ['participants_count', 'injured_count', 'dead_count', 'vehicles_count']
    assert frame.source_row.is_unique and (frame[counts] >= 0).all().all()
    assert (frame.injured_count + frame.dead_count <= frame.participants_count).all()
    missing = frame[['longitude_raw', 'latitude_raw']].isna().any(axis=1)
    zeros = frame.longitude_raw.eq(0) & frame.latitude_raw.eq(0)
    return dict(rows=len(frame), full_year_rows=int(frame.year.between(2015, 2025).sum()),
                duplicate_nonempty_ids=int(frame.loc[frame.id.notna(), 'id'].duplicated().sum()),
                missing=frame.isna().sum().to_dict(), maximum=frame[counts].max().to_dict(),
                category_counts=frame.category.value_counts().to_dict(),
                year_counts={str(year): int(n) for year, n in frame.groupby('year').size().items()},
                injured_distribution=dict(at_most_one=float(frame.injured_count.le(1).mean()),
                                          at_most_two=float(frame.injured_count.le(2).mean())),
                map=dict(valid=int(frame.map_valid.sum()), excluded=int((~frame.map_valid).sum()),
                         missing=int(missing.sum()), zeros=int(zeros.sum()),
                         other=int((~frame.map_valid & ~missing & ~zeros).sum()),
                         corrected_swaps=int(frame.coordinates_swapped.sum())))


def main():
    frame = records_to_frame(load_records(SOURCE))
    with SOURCE.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    source_metadata = json.loads(SOURCE.with_suffix('.source.json').read_text())
    assert source_metadata['sha256'] == digest
    cache, figures = [ROOT / 'solution' / name for name in ('cache', 'figures')]
    for directory in (cache, figures):
        directory.mkdir(exist_ok=True)
    print('Считаем гипотезы, PCA и два t-SNE...', flush=True)
    projected = compute_projections(frame)
    sample = projected.pop('data')
    columns = ['participants_count', 'vehicles_count', 'category_grouped', 'light']
    groups = sample.groupby(columns)
    profile = groups.size().idxmax()
    common = groups.get_group(profile)
    projected['metadata']['unique_feature_rows_sample'] = len(groups)
    projected['summaries']['common_profile'] = dict(profile=dict(zip(columns, profile)), n=len(common),
                                                   outcomes=common.severity.value_counts().to_dict())
    sample.to_csv(cache / 'projections.csv', index=False)
    result = dict(source_sha256=digest,
                  source={key: source_metadata[key] for key in ('source_page', 'source_api', 'region',
                          'downloaded_at_utc', 'requested_from', 'requested_to')},
                  data=data_summary(frame), hypotheses=run_hypotheses(frame), projections=projected)
    charts.export_before_after(frame, figures)
    grid = charts.grid_map(frame).data[0]
    counts = np.asarray(grid.z, dtype=int)
    assert counts.sum() == frame.map_valid.sum()
    corners = grid.geojson['features'][int(counts.argmax())]['geometry']['coordinates'][0][:4]
    result['data']['map'].update(occupied_cells=len(counts), largest_cell_count=int(counts.max()),
                               largest_cell_share=float(counts.max() / counts.sum()),
                               largest_cell_center_lon_lat=np.mean(corners, axis=0).tolist(),
                               top10_cells_share=float(np.sort(counts)[-10:].sum() / counts.sum()))
    (ROOT / 'solution/results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('Сохранены results.json, проекции и PNG до/после.')


if __name__ == '__main__':
    main()
