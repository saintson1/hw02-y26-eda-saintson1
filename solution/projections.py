"""PCA и два t-SNE на одних ДТП. последствия используются только для цвета."""
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

SEED = 42
NUMERIC = ["participants_count", "vehicles_count"]
SEVERITY_COLORS = {"С погибшими": "#B33440", "С ранеными": "#3579B8",
                   "Без пострадавших": "#8D939C", "Неизвестно": "#A38B69"}


def _labels(values):
    return values.astype("string").str.strip().replace("", pd.NA).fillna("Не указано")


def compute_projections(frame):
    """Вернуть data/metadata/summaries. считать по фиксированной выборке 2015-2025."""
    columns = ["source_row", "year", "category", "light", "severity", "fatal", *NUMERIC]
    full = frame.loc[frame.year.between(2015, 2025), columns].copy()
    for name in ("category", "light"):
        full[name] = _labels(full[name])
    counts = full.category.value_counts()
    rare = counts[(counts < 100) & (counts.index != "Не указано")].index
    full["category_grouped"] = full.category.where(~full.category.isin(rare), "Другие")
    full[NUMERIC] = full[NUMERIC].astype(float)
    sample = (full.sort_values("source_row", kind="stable")
              .sample(n=min(2500, len(full)), random_state=SEED)
              .sort_values("source_row").reset_index(drop=True))
    sample["severity"] = _labels(sample.severity).replace("Не указано", "Неизвестно")
    medians = sample[NUMERIC].median()
    scaler = StandardScaler()
    encoder = OneHotEncoder(sparse_output=False)
    with threadpool_limits(limits=1):
        # OHE остаётся 0/1: стандартизация усилила бы редкие уровни.
        matrix = np.column_stack([scaler.fit_transform(sample[NUMERIC].fillna(medians)),
                                  encoder.fit_transform(sample[["category_grouped", "light"]])])
        pca = PCA(n_components=2, svd_solver="full")
        sample[["pca1", "pca2"]] = pca.fit_transform(matrix)
        parameters = {}
        for perplexity in (15, 40):
            params = dict(perplexity=perplexity, init="pca", learning_rate="auto",
                          max_iter=1000, random_state=SEED)
            sample[[f"tsne{perplexity}_1", f"tsne{perplexity}_2"]] = TSNE(
                n_components=2, n_jobs=1, **params).fit_transform(matrix)
            parameters[str(perplexity)] = params
    metadata = {
        "seed": SEED, "period_rows": len(full), "sample_rows": len(sample),
        "input_features": [*NUMERIC, "category", "light"],
        "features": [*NUMERIC, *encoder.get_feature_names_out(["category_grouped", "light"]).tolist()],
        "medians": medians.to_dict(),
        "scaling": {"columns": NUMERIC, "mean": scaler.mean_.tolist(),
                    "scale": scaler.scale_.tolist(), "ohe_scaled": False},
        "pca": {"explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "retained_variance": float(pca.explained_variance_ratio_.sum()),
                "components": pca.components_.tolist()}, "tsne": parameters}
    low, high = sample.pca1.quantile([0.25, 0.75])
    quartiles = {}
    for name, group in (("low", sample[sample.pca1 <= low]), ("high", sample[sample.pca1 >= high])):
        quartiles[name] = {"n": len(group), "participants_mean": float(group.participants_count.mean()),
                           "vehicles_mean": float(group.vehicles_count.mean()),
                           "pedestrian_share": float(group.category.eq("Наезд на пешехода").mean()),
                           "collision_share": float(group.category.eq("Столкновение").mean())}
    return {"data": sample, "metadata": metadata, "summaries": {"pca_quartiles": quartiles}}


def _scatter(frame, **kwargs):
    figure = px.scatter(
        frame, color="severity", color_discrete_map=SEVERITY_COLORS,
        category_orders={"severity": list(SEVERITY_COLORS)},
        hover_data=["source_row", "category", "light", "fatal", *NUMERIC],
        labels={"severity": "Последствия", "source_row": "Строка источника", "category": "Тип ДТП",
                "light": "Освещение", "fatal": "Есть погибшие", "participants_count": "Участники",
                "vehicles_count": "Транспортные средства"}, **kwargs)
    figure.update_traces(marker={"size": 5, "opacity": 0.65})
    figure.update_layout(template="plotly_white", legend_title_text="Последствия", height=580)
    return figure


def pca_figure(result):
    evr = result["metadata"]["pca"]["explained_variance_ratio"]
    figure = _scatter(result["data"], x="pca1", y="pca2",
                      title=f"11. PCA, {len(result['data'])} ДТП, сохранено {sum(evr):.1%} дисперсии X")
    figure.update_xaxes(title=f"PC1, {evr[0]:.1%} дисперсии X")
    figure.update_yaxes(title=f"PC2, {evr[1]:.1%} дисперсии X")
    return figure


def tsne_figure(result):
    panels = [result["data"].assign(x=lambda f: f[f"tsne{p}_1"], y=lambda f: f[f"tsne{p}_2"],
                                  setting=f"perplexity={p}") for p in (15, 40)]
    figure = _scatter(pd.concat(panels, ignore_index=True), x="x", y="y", facet_col="setting",
                      title="12. t-SNE, те же ДТП и признаки, seed=42<br>"
                            "<sup>Оси без физических единиц. пятна не доказывают существование кластеров</sup>")
    figure.update_xaxes(matches=None, title="t-SNE 1", showticklabels=False)
    figure.update_yaxes(matches=None, title="t-SNE 2", showticklabels=False)
    figure.for_each_annotation(lambda a: a.update(text=a.text.replace("setting=", "")))
    return figure
