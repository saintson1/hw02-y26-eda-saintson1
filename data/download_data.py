"""Download the assigned Nizhny Novgorod dataset: python data/download_data.py."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlencode
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
API = "https://dtpstat-api.dexoon.workers.dev/api/upstream/dtp/geojson"
REGION_ID = 14
REGION_NAME = "Нижегородская область"


def main():
    target = ROOT / "data/raw/nizhegorodskaia-oblast.geojson"
    if target.exists():
        print(f"Уже существует: {target}. Файл не перезаписан. для новой версии переместите его вместе с .source.json.")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    annual = []
    dates, identifiers = [], []
    count = 0
    with TemporaryDirectory(dir=target.parent) as temporary:
        partial = Path(temporary) / "region.geojson"
        with partial.open("w", encoding="utf-8") as output:
            output.write('{"type":"FeatureCollection","features":[')
            # The API limits each request to at most 366 days.
            for year in range(2015, 2027):
                start = f"{year}-01-01T00:00:00"
                end = f"{year}-12-31T23:59:59" if year < 2026 else "2026-01-31T23:59:59"
                url = API + "?" + urlencode({"REGION": REGION_ID, "DATE_TIME_FROM": start, "DATE_TIME_TO": end})
                with urlopen(url, timeout=120) as response:
                    payload = response.read()
                collection = json.loads(payload)
                features = collection.get("features") if isinstance(collection, dict) else None
                if not isinstance(features, list) or not features or collection.get("type") != "FeatureCollection":
                    raise ValueError(f"Нет корректного непустого FeatureCollection за {year}")
                for feature in features:
                    props = feature.get("properties") if isinstance(feature, dict) else None
                    if not isinstance(props, dict) or feature.get("type") != "Feature":
                        raise ValueError(f"Некорректный объект ДТП за {year}")
                    date, regions = props.get("DATE_TIME"), props.get("REGIONS")
                    if (not isinstance(date, str) or not isinstance(regions, list)
                            or REGION_NAME not in regions or not start <= date <= end):
                        raise ValueError(f"Запись не соответствует запрошенному региону/периоду: {year}")
                    if count:
                        output.write(",")
                    json.dump(feature, output, ensure_ascii=False, separators=(",", ":"))
                    count += 1
                    dates.append(date)
                    identifiers.append(props.get("EM_NUMBER"))
                annual.append({"year": year, "url": url, "records": len(features),
                               "response_bytes": len(payload),
                               "response_sha256": hashlib.sha256(payload).hexdigest()})
                print(f"{year}: {len(features):,} ДТП", flush=True)
            output.write("]}\n")
        partial.replace(target)
    nonempty_ids = [value for value in identifiers if value is not None]
    with target.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    metadata = {
        "source_page": "https://dtp-stat.ru/download/", "source_api": API,
        "region": REGION_NAME, "region_id": REGION_ID,
        "download_started_utc": started,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_from": "2015-01-01T00:00:00", "requested_to": "2026-01-31T23:59:59",
        "min_datetime": min(dates), "max_datetime": max(dates),
        "records": count, "bytes": target.stat().st_size, "sha256": digest,
        "duplicate_nonempty_ids": len(nonempty_ids) - len(set(nonempty_ids)),
        "annual_downloads": annual,
        "note": "Live API snapshot with fixed period",
    }
    target.with_suffix(".source.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Сохранено {count:,} записей: {target}\nSHA-256: {metadata['sha256']}")


if __name__ == "__main__":
    main()
