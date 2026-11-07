"""Интерактивное исследование ДТП: make run."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

import charts
from data_loading import load_records, records_to_frame
from projections import compute_projections, pca_figure, tsne_figure
from stats import run_hypotheses

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/raw/nizhegorodskaia-oblast.geojson'
st.set_page_config(page_title='ДТП, Нижегородская область', layout='wide')


@st.cache_data(show_spinner='Загрузка данных...')
def read_frame(path, modified_ns):
    with open(path, 'rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    return records_to_frame(load_records(Path(path))), digest


@st.cache_data(show_spinner='Расчёт гипотез и проекций...')
def fixed_analysis(_frame, digest):
    results_path = ROOT / 'solution/results.json'
    projection_path = ROOT / 'solution/cache/projections.csv'
    if results_path.exists() and projection_path.exists():
        results = json.loads(results_path.read_text())
        if results['source_sha256'] == digest:
            return results['hypotheses'], dict(results['projections'], data=pd.read_csv(projection_path))
    return run_hypotheses(_frame), compute_projections(_frame)


def show(figure):
    if figure is None:
        st.info('Недостаточно данных для сравнения.')
    else:
        st.plotly_chart(figure, width='stretch')


def reset_filters():
    for key in list(st.session_state):
        if key.startswith('filter_'):
            del st.session_state[key]


def main():
    st.title('ДТП в Нижегородской области')
    st.caption('saintson1, одно наблюдение - одно зарегистрированное ДТП')
    if not SOURCE.exists():
        st.info('Загрузите данные: .venv/bin/python data/download_data.py')
        return
    try:
        frame, digest = read_frame(str(SOURCE), SOURCE.stat().st_mtime_ns)
    except (ValueError, OSError, TypeError) as error:
        st.error(str(error))
        return
    dated = frame.dropna(subset=['datetime'])
    if dated.empty:
        st.info('Нет записей с распознанной датой.')
        return
    first, last = dated.datetime.min().date(), dated.datetime.max().date()
    st.sidebar.button('Сбросить фильтры', on_click=reset_filters)
    period = st.sidebar.date_input('Период', (first, last), min_value=first, max_value=last, key='filter_period')
    if len(period) != 2:
        st.info('Выберите обе границы периода.')
        return
    start, end = period
    current = dated[(dated.datetime >= pd.Timestamp(start)) &
                    (dated.datetime < pd.Timestamp(end) + pd.Timedelta(days=1))]
    for column, title, all_label in [('region', 'Территория', 'Все территории'),
                                      ('category', 'Тип ДТП', 'Все типы ДТП')]:
        if not st.sidebar.checkbox(all_label, value=True, key=f'filter_all_{column}'):
            selected = st.sidebar.multiselect(title, sorted(frame[column].dropna().unique()), key=f'filter_{column}')
            current = current[current[column].isin(selected)]
    if current.empty:
        st.info('В выбранном срезе нет ДТП.')
        return
    st.caption(f'{start:%d.%m.%Y}-{end:%d.%m.%Y}, графики и карты учитывают фильтры')
    values = [('ДТП', len(current)), ('Раненые', current.injured_count.sum()),
              ('ДТП с погибшими', current.fatal.sum()), ('Непригодные координаты', (~current.map_valid).sum())]
    for column, (label, value) in zip(st.columns(4), values):
        column.metric(label, f'{int(value):,}')
    overview, questions, geography, hypotheses, projections, data = st.tabs(
        ['Обзор', 'Вопросы исследования', 'Карта', 'Три гипотезы', 'PCA и t-SNE', 'Данные'])
    with overview:
        show(charts.monthly(current, start, end))
        st.caption('2026 год содержит только январь. крайние месяцы фильтра могут быть неполными.')
        for figure in [charts.categories(current), charts.injured_ecdf(current), charts.hour_weekday(current)]:
            show(figure)
    with questions:
        show(charts.night_by_weekday(current))
        st.write('После сравнения ночной доли я отдельно сравнил число раненых в столкновениях.')
        show(charts.night_collision_injuries(current))
        show(charts.pedestrian_seasons(current[current.year.between(2015, 2025)]))
        st.caption('Сезонное сравнение исключает неполный 2026 год.')
        show(charts.participants_injured(current))
    with geography:
        backdrop = st.checkbox('Подложка OpenStreetMap', value=True)
        show(charts.point_map(current, backdrop))
        show(charts.grid_map(current, backdrop=backdrop))
        st.caption('Сетка 5 на 5 км, EPSG:32638. (c) OpenStreetMap contributors - https://www.openstreetmap.org/copyright')
    checked, projected = fixed_analysis(frame, digest)
    with hypotheses:
        st.caption(f'2015-2025, {checked["full_year_rows"]:,} ДТП, фильтры не применяются')
        for item in checked['results']:
            st.markdown(f'**{item["id"]}. {item["question"]}**')
            st.write(f'{item["group_a"]}: n={item["n_a"]:,}. {item["group_b"]}: n={item["n_b"]:,}.')
            st.write(f'{item["test"]}: p={item["p_value"]:.4g}. Разность {item["effect"]:+.3f} {item["unit"]}. '
                     f'95% доверительный интервал от {item["ci_low"]:+.3f} до {item["ci_high"]:+.3f}.')
            st.caption('Я отвергаю H0.' if item['reject'] else 'Я не отвергаю H0.')
        st.caption('H0 - равенство долей или средних, H1 - различие. Уровень значимости 0,05. Bootstrap: 5000 повторов '
                   'целых ДТП. Перестановки: 9999. p=(b+1)/10000. Ночь: 22:00-05:59, выходные: сб-вс.')
    with projections:
        metadata = projected['metadata']
        st.caption(f'{metadata["sample_rows"]} ДТП 2015-2025, seed=42, фильтры не применяются')
        st.write('Я стандартизировал числа участников и ТС, тип и освещение закодировал OHE. Последствия использовал только для цвета.')
        show(pca_figure(projected))
        with st.expander('Коэффициенты PCA'):
            st.dataframe(pd.DataFrame(metadata['pca']['components'], index=['PC1', 'PC2'], columns=metadata['features']).T)
        show(tsne_figure(projected))
        st.caption('Perplexity 15/40, init=pca, learning_rate=auto, 1000 итераций. Пятна не доказывают кластеры.')
    with data:
        st.markdown('[Источник: "Карта ДТП"](https://dtp-stat.ru/download/). Период: 01.01.2015-31.01.2026.')
        st.write(f'Я исправил {int(frame.coordinates_swapped.sum())} перепутанных пар координат. '
                 f'Из карт исключил {int((~frame.map_valid).sum())} записей.')
        st.write('Я проверил долготу от 41 до 49 и широту от 54 до 59. Последствия определял по числу погибших и раненых.')
        st.caption('Полный отчёт: solution/analysis.md. Без данных о потоке нельзя оценить риск поездки.')
        st.dataframe(current[['id', 'datetime', 'region', 'category', 'injured_count', 'dead_count']].head(100), hide_index=True)


if __name__ == '__main__':
    main()
