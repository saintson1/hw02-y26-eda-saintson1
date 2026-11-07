import json
from pathlib import Path

import pandas as pd


def load_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict) and payload.get("type") == "FeatureCollection":
        records = payload.get("features")
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError("Ожидается GeoJSON FeatureCollection или список записей.")
    if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
        raise ValueError("Выгрузка должна содержать список объектов JSON.")
    return records


def records_to_frame(records: list[dict]) -> pd.DataFrame:
    aliases = {
        "id": "EM_NUMBER", "datetime": "DATE_TIME", "category": "EM_TYPE",
        "light": "LIGHT_TYPE",
        "participants_count": "PERS_AMOUNT", "injured_count": "SUFFER_AMOUNT",
        "dead_count": "LOST_AMOUNT", "vehicles_count": "VEH_AMOUNT",
    }
    rows = []
    for index, record in enumerate(records):
        props = record.get("properties") if record.get("type") == "Feature" else record
        props = props if isinstance(props, dict) else {}
        row = {key: props.get(key, props.get(upper)) for key, upper in aliases.items()}
        row["region"] = props.get("region") or props.get("PLACE") or props.get("NP")
        row["source_row"] = index
        geometry = record.get("geometry")
        geometry = geometry if isinstance(geometry, dict) else {}
        if not geometry:
            geometry = props.get("POINT")
            geometry = geometry if isinstance(geometry, dict) else {}
        coords = geometry.get("coordinates")
        if geometry.get("type") == "Point" and isinstance(coords, list) and len(coords) >= 2:
            row["longitude"], row["latitude"] = coords[:2]
        else:
            point = props.get("point")
            point = point if isinstance(point, dict) else {}
            row["longitude"], row["latitude"] = point.get("long"), point.get("lat")
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce", format="mixed")
    for name in ("longitude", "latitude", "participants_count", "injured_count", "dead_count", "vehicles_count"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame["longitude_raw"] = frame["longitude"]
    frame["latitude_raw"] = frame["latitude"]
    original_inside = frame["longitude"].between(41, 49) & frame["latitude"].between(54, 59)
    frame["coordinates_swapped"] = (
        ~original_inside & frame["latitude"].between(41, 49) & frame["longitude"].between(54, 59))
    swapped = frame["coordinates_swapped"]
    frame.loc[swapped, ["longitude", "latitude"]] = frame.loc[swapped, ["latitude", "longitude"]].to_numpy()
    frame["severity"] = "Неизвестно"
    frame.loc[frame["dead_count"].eq(0) & frame["injured_count"].eq(0), "severity"] = "Без пострадавших"
    frame.loc[frame["dead_count"].eq(0) & frame["injured_count"].gt(0), "severity"] = "С ранеными"
    frame.loc[frame["dead_count"].gt(0), "severity"] = "С погибшими"
    frame["fatal"] = frame["dead_count"].gt(0)
    for name in ("year", "month", "hour", "day_of_week"):
        attribute = "dayofweek" if name == "day_of_week" else name
        frame[name] = getattr(frame["datetime"].dt, attribute).astype("Int64")
    frame["weekend"] = frame["day_of_week"].ge(5)
    frame["night"] = frame["hour"].ge(22) | frame["hour"].lt(6)
    frame["season"] = frame["month"].map({
        12: "зима", 1: "зима", 2: "зима", 3: "весна", 4: "весна", 5: "весна",
        6: "лето", 7: "лето", 8: "лето", 9: "осень", 10: "осень", 11: "осень",
    })
    frame["global_valid"] = (
        frame["longitude"].between(-180, 180)
        & frame["latitude"].between(-90, 90)
    )
    # This rectangle checks plausibility, not the administrative region boundary.
    frame["inside_region_box"] = frame["longitude"].between(41, 49) & frame["latitude"].between(54, 59)
    frame["map_valid"] = frame["global_valid"] & frame["inside_region_box"]
    return frame
