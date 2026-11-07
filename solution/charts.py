"""Графики текущего среза. одно наблюдение - одно ДТП."""
from pathlib import Path
from textwrap import wrap

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pyproj import Transformer

BLUE, ORANGE = '#3579B8', '#C18A16'
SEVERITY_COLORS = {'С ранеными': BLUE, 'С погибшими': '#B33440',
                   'Без пострадавших': '#779184', 'Неизвестно': '#8D939C'}
WEEKDAYS = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']


def _layout(fig, title, height=440):
    fig.update_layout(template='plotly_white', title=dict(text=title, x=0), height=height,
                      margin=dict(l=35, r=25, t=65, b=55), font=dict(color='#263445'))
    return fig


def _category_labels(names, separator):
    labels = [separator.join(wrap(str(name), width=45)) for name in names]
    lines = np.array([label.count(separator) + 1 for label in labels])
    return labels, np.cumsum(lines) - lines / 2, int(lines.sum())


def monthly(frame, start, end):
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    dates = frame['datetime'].dropna()
    dates = dates[(dates >= start) & (dates < end + pd.Timedelta(days=1))]
    if dates.empty:
        return None
    months = pd.date_range(start.to_period('M').start_time, end.to_period('M').start_time, freq='MS')
    counts = dates.dt.to_period('M').value_counts().reindex(months.to_period('M'), fill_value=0)
    status = ['Неполный месяц' if month < start or month + pd.offsets.MonthEnd(0) > end
              else 'Полный месяц периода' for month in months]
    fig = go.Figure(go.Scatter(x=months, y=counts, mode='lines+markers', line_color=BLUE,
                    customdata=status, hovertemplate='%{x|%m.%Y}: %{y} ДТП<br>%{customdata}<extra></extra>'))
    fig.update_xaxes(title='Месяц')
    fig.update_yaxes(title='Число ДТП', rangemode='tozero')
    return _layout(fig, f'1. ДТП по месяцам, N = {len(dates):,}')


def categories(frame):
    if frame.empty:
        return None
    counts = frame['category'].fillna('Неизвестно').value_counts().sort_values(kind='stable')
    labels, positions, lines = _category_labels(counts.index, '<br>')
    shares = counts.to_numpy() / len(frame) * 100
    fig = go.Figure(go.Bar(x=counts, y=positions, orientation='h', marker_color=BLUE, width=0.7,
                    text=[f'{n:,} ({share:.1f}%)' for n, share in zip(counts, shares)],
                    textposition='outside', cliponaxis=False, hovertext=counts.index,
                    customdata=shares, hovertemplate='%{hovertext}<br>%{x} ДТП, %{customdata:.1f}%<extra></extra>'))
    fig.update_xaxes(title='Число ДТП', range=[0, float(counts.max()) * 1.27])
    fig.update_yaxes(tickvals=positions, ticktext=labels, automargin=True, range=[0, lines])
    return _layout(fig, f'2. Типы ДТП, N = {len(frame):,}, % от всех ДТП среза', max(480, lines * 22 + 120))


def injured_ecdf(frame):
    values = frame['injured_count'].dropna()
    values = values[np.isfinite(values)]
    if values.empty:
        return None
    counts = values.value_counts().sort_index()
    cumulative = np.r_[0, counts.cumsum().to_numpy()]
    fig = go.Figure(go.Scatter(x=np.r_[min(0, values.min()) - 0.5, counts.index],
                    y=cumulative / len(values) * 100, mode='lines+markers',
                    line=dict(color=BLUE, shape='hv'), customdata=cumulative,
                    hovertemplate='Раненых не больше %{x}: %{customdata} ДТП (%{y:.1f}%)<extra></extra>'))
    fig.update_xaxes(title='Раненых на ДТП', range=[min(0, values.min()) - 0.5, values.max() + 0.5],
                     dtick=1 if values.max() <= 30 else None)
    fig.update_yaxes(title='Доля ДТП с числом раненых не больше x, %', range=[0, 102])
    return _layout(fig, f'3. ECDF раненых, n = {len(values):,} ДТП с известным числом раненых')


def hour_weekday(frame):
    valid = frame.dropna(subset=['day_of_week', 'hour'])
    if valid.empty:
        return None
    counts = pd.crosstab(valid['day_of_week'], valid['hour']).reindex(index=range(7), columns=range(24), fill_value=0)
    totals = counts.sum(axis=1).to_numpy()[:, None]
    shares = np.divide(counts.to_numpy() * 100, totals, out=np.full((7, 24), np.nan), where=totals > 0)
    custom = np.stack([counts.to_numpy(), np.broadcast_to(totals, (7, 24))], axis=2)
    fig = go.Figure(go.Heatmap(x=list(range(24)), y=WEEKDAYS, z=shares, customdata=custom,
                    colorscale='Blues', zmin=0, colorbar=dict(title='% ДТП<br>дня недели'), hoverongaps=False,
                    hovertemplate='%{y}, %{x}:00, %{customdata[0]} / %{customdata[1]} ДТП (%{z:.2f}%)<extra></extra>'))
    fig.update_xaxes(title='Час', dtick=1)
    fig.update_yaxes(title='День недели', autorange='reversed')
    return _layout(fig, f'4. Распределение по часам, N = {len(valid):,}, % внутри дня недели')


def night_by_weekday(frame):
    valid = frame.dropna(subset=['day_of_week', 'night'])
    if valid.empty:
        return None
    groups = valid.groupby('day_of_week')['night'].agg(['sum', 'size']).reindex(range(7))
    fig = go.Figure(go.Scatter(x=WEEKDAYS, y=groups['sum'] / groups['size'] * 100, mode='markers',
                    marker=dict(color=[BLUE] * 5 + [ORANGE] * 2, size=12), customdata=groups.to_numpy(),
                    hovertemplate='%{x}: %{customdata[0]} / %{customdata[1]} ДТП (%{y:.1f}%)<extra></extra>'))
    fig.update_yaxes(title='Доля ночных ДТП, %', range=[0, 100])
    return _layout(fig, f'5. Ночь 22:00-05:59, N = {len(valid):,}, доля от ДТП каждого дня недели')


def night_collision_injuries(frame):
    selected = frame[frame['night'] & frame['category'].eq('Столкновение')].dropna(subset=['injured_count', 'weekend'])
    groups = [selected.loc[selected['weekend'].eq(value), 'injured_count'] for value in (False, True)]
    if any(group.empty for group in groups):
        return None
    fig = go.Figure()
    for label, color, values in zip(('Будни', 'Выходные'), (BLUE, ORANGE), groups):
        fig.add_trace(go.Box(y=values, name=f'{label}, n = {len(values):,}', marker_color=color,
                            boxmean=True, boxpoints='outliers', hovertemplate='Раненых: %{y}<extra>%{fullData.name}</extra>'))
    fig.update_yaxes(title='Раненых на ДТП', rangemode='tozero')
    fig.update_layout(showlegend=False)
    return _layout(fig, f'6. Ночные столкновения, n = {len(selected):,}')


def pedestrian_seasons(frame):
    selected = frame[frame['season'].isin(['зима', 'лето'])]
    total = selected.groupby('season').size().reindex(['зима', 'лето'], fill_value=0)
    if (total == 0).any():
        return None
    counts = selected['category'].eq('Наезд на пешехода').groupby(selected['season']).sum().reindex(total.index)
    fig = go.Figure(go.Bar(x=['Зима', 'Лето'], y=counts / total * 100, marker_color=[BLUE, ORANGE],
                    text=[f'{n:,} / {all_events:,}' for n, all_events in zip(counts, total)],
                    textposition='outside', cliponaxis=False, customdata=np.column_stack([counts, total]),
                    hovertemplate='%{x}: %{customdata[0]} / %{customdata[1]} ДТП (%{y:.1f}%)<extra></extra>'))
    fig.update_yaxes(title='Наезды на пешеходов / все ДТП сезона, %', rangemode='tozero')
    return _layout(fig, f'7. Пешеходные ДТП по сезонам, N = {len(selected):,}')


def participants_injured(frame):
    valid = frame.dropna(subset=['participants_count', 'injured_count'])
    if valid.empty:
        return None
    counts = valid.groupby(['participants_count', 'injured_count']).size().reset_index(name='count')
    fig = px.scatter(counts, x='participants_count', y='injured_count', size='count', size_max=40,
                     labels={'participants_count': 'Участников на ДТП', 'injured_count': 'Раненых на ДТП', 'count': 'ДТП'},
                     color_discrete_sequence=[BLUE])
    fig.update_xaxes(rangemode='tozero')
    fig.update_yaxes(rangemode='tozero')
    return _layout(fig, f'8. Участники и раненые, n = {len(valid):,}, площадь точки = число ДТП')


def point_map(frame, backdrop=True):
    points = frame[frame['map_valid']].copy()
    if points.empty:
        return None
    total = len(points)
    points = points.sample(min(total, 5000), random_state=42)
    points['severity'] = points['severity'].fillna('Неизвестно')
    fig = px.scatter_map(points, lat='latitude', lon='longitude', color='severity',
                         color_discrete_map=SEVERITY_COLORS, hover_data=['datetime', 'category'],
                         labels={'severity': 'Последствия', 'datetime': 'Дата', 'category': 'Тип ДТП'},
                         opacity=0.55, zoom=6, map_style='open-street-map' if backdrop else 'white-bg',
                         center=dict(lat=float(points['latitude'].median()), lon=float(points['longitude'].median())))
    return _layout(fig, f'9. Точки ДТП, показано {len(points):,} / {total:,}', 600)


def grid_map(frame, step_m=5000, backdrop=True):
    if not np.isfinite(step_m) or step_m <= 0:
        raise ValueError('Размер ячейки должен быть положительным числом метров.')
    points = frame[frame['map_valid']]
    if points.empty:
        return None
    forward = Transformer.from_crs(4326, 32638, always_xy=True)
    reverse = Transformer.from_crs(32638, 4326, always_xy=True)
    x, y = forward.transform(points['longitude'].to_numpy(), points['latitude'].to_numpy())
    cells = pd.DataFrame({'x': np.floor(x / step_m).astype(int), 'y': np.floor(y / step_m).astype(int)})
    counts = cells.groupby(['x', 'y']).size().reset_index(name='count')
    features = []
    for index, row in counts.iterrows():
        left, bottom = row['x'] * step_m, row['y'] * step_m
        lon, lat = reverse.transform([left, left + step_m, left + step_m, left, left],
                                     [bottom, bottom, bottom + step_m, bottom + step_m, bottom])
        features.append(dict(type='Feature', id=str(index), properties={},
                             geometry=dict(type='Polygon', coordinates=[list(map(list, zip(lon, lat)))])))
    fig = go.Figure(go.Choroplethmap(geojson=dict(type='FeatureCollection', features=features),
                    locations=[str(i) for i in counts.index], z=counts['count'], colorscale='Blues', zmin=0,
                    colorbar=dict(title='Число ДТП'), marker=dict(opacity=0.7, line=dict(width=0.5, color='#52677A')),
                    hovertemplate='ДТП в ячейке: %{z}<extra></extra>'))
    fig.update_layout(map=dict(style='open-street-map' if backdrop else 'white-bg', zoom=6,
                               center=dict(lat=float(points['latitude'].median()), lon=float(points['longitude'].median()))))
    return _layout(fig, f'10. Сетка {step_m / 1000:g}  на  {step_m / 1000:g} км, N = {len(points):,}', 600)


def export_before_after(frame, directory):
    """Два оформления тех же частот, без объединения категорий."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if frame.empty:
        return None
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    counts = frame['category'].fillna('Неизвестно').value_counts(sort=False)
    paths = []
    for before in (True, False):
        values = counts if before else counts.sort_values(kind='stable')
        labels, positions, lines = _category_labels(values.index, '\n')
        fig = Figure(figsize=(12, 7 if before else max(6, lines * 0.25)))
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        if before:
            ax.bar(values.index, values.to_numpy(), color=BLUE)
            ax.tick_params(axis='x', rotation=70, labelsize=8)
            ax.set_ylabel('Число ДТП')
            ax.set_ylim(bottom=0)
            fig.subplots_adjust(bottom=0.45)
        else:
            bars = ax.barh(positions, values.to_numpy(), color=BLUE, height=0.7)
            ax.set_yticks(positions, labels)
            ax.set_ylim(0, lines)
            ax.bar_label(bars, labels=[f'{n:,} ({n / len(frame):.1%})' for n in values], padding=5)
            ax.set_xlim(0, values.max() * 1.3)
            ax.set_xlabel('Число ДТП')
            fig.tight_layout()
        ax.set_title(f'Типы ДТП, N = {len(frame):,}. доли от всех ДТП среза')
        path = directory / ('categories_before.png' if before else 'categories_after.png')
        fig.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0.2)
        paths.append(path)
    return tuple(paths)
