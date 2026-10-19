
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from shapely.geometry import Polygon, mapping

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)


INVALID_SCL = [0, 1, 3, 7, 8, 9, 10, 11]
FEATURE_NAMES = [
    "B2_Blue", "B3_Green", "B4_Red", "B8_NIR",
    "B11_SWIR1", "B12_SWIR2", "NDVI", "BSI"
]


def _tag(element):
    return element.tag.rsplit("}", 1)[-1]


def _coords(text):
    points = []
    for item in text.strip().split():
        values = item.split(",")
        if len(values) >= 2:
            try:
                points.append((float(values[0]), float(values[1])))
            except ValueError:
                pass
    if len(points) >= 3 and points[0] != points[-1]:
        points.append(points[0])
    return points


def read_kmz(path):
    """Read Polygon geometries from a KMZ archive."""
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist()
                 if n.lower().endswith(".kml")]
        if not names:
            raise ValueError(f"No KML inside {path.name}")
        kml_name = next(
            (n for n in names if n.lower().endswith("doc.kml")),
            names[0]
        )
        root = ET.fromstring(archive.read(kml_name))

    polygons = []
    for element in root.iter():
        if _tag(element) != "Polygon":
            continue

        outer = None
        holes = []
        for boundary in element:
            boundary_name = _tag(boundary)
            if boundary_name not in ("outerBoundaryIs", "innerBoundaryIs"):
                continue

            ring = None
            for child in boundary.iter():
                if _tag(child) == "coordinates" and child.text:
                    ring = _coords(child.text)
                    break

            if ring and len(ring) >= 4:
                if boundary_name == "outerBoundaryIs":
                    outer = ring
                else:
                    holes.append(ring)

        if outer:
            polygon = Polygon(outer, holes=holes)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if not polygon.is_empty:
                if polygon.geom_type == "Polygon":
                    polygons.append(polygon)
                elif polygon.geom_type == "MultiPolygon":
                    polygons.extend(polygon.geoms)

    if not polygons:
        raise ValueError(f"No valid polygons in {path.name}")
    return polygons


def read_features(path):
    """Read B2, B3, B4, B8, B11, B12, SCL in that order."""
    with rasterio.open(path) as src:
        if src.count < 7:
            raise ValueError(
                "Raster must contain 7 bands: B2,B3,B4,B8,B11,B12,SCL."
            )
        data = src.read(out_dtype="float32")
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    b2, b3, b4, b8, b11, b12, scl = data[:7]

    valid = np.all(np.isfinite(data[:6]), axis=0)
    valid &= np.isfinite(scl)
    if nodata is not None:
        valid &= np.all(data[:6] != nodata, axis=0)
    valid &= ~np.isin(scl, INVALID_SCL)

    def ratio(a, b):
        result = np.full(a.shape, np.nan, dtype="float32")
        ok = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-8)
        result[ok] = a[ok] / b[ok]
        return result

    ndvi = ratio(b8 - b4, b8 + b4)
    bsi = ratio((b11 + b4) - (b8 + b2),
                (b11 + b4) + (b8 + b2))

    features = np.stack(
        [b2, b3, b4, b8, b11, b12, ndvi, bsi]
    ).astype("float32")

    valid &= np.all(np.isfinite(features), axis=0)
    return features, valid, profile, transform, crs


def polygon_mask(kmz_path, shape, transform, crs):
    geometries = []
    for polygon in read_kmz(kmz_path):
        geom = transform_geom(
            "EPSG:4326", crs, mapping(polygon), precision=-1
        )
        geometries.append((geom, 1))

    return rasterize(
        geometries, out_shape=shape, transform=transform,
        fill=0, default_value=1, all_touched=False, dtype="uint8"
    ).astype(bool)


def train_model(raster_path, labels_dir):
    """Train from quarry and four non-quarry KMZ labels."""
    labels_dir = Path(labels_dir)
    features, valid, profile, transform, crs = read_features(raster_path)
    shape = valid.shape

    quarry = polygon_mask(
        labels_dir / "Known_Quarry_2021.kmz", shape, transform, crs
    )

    negatives = np.zeros(shape, dtype=bool)
    for filename in [
        "NonQuarry_Water_2021.kmz",
        "NonQuarry_RiverSediment_2021.kmz",
        "NonQuarry_Roads_2021.kmz",
        "NonQuarry_Agriculture_2021.kmz",
    ]:
        negatives |= polygon_mask(
            labels_dir / filename, shape, transform, crs
        )

    overlap = quarry & negatives
    quarry &= ~overlap & valid
    negatives &= ~overlap & valid

    if quarry.sum() < 10 or negatives.sum() < 10:
        raise ValueError(
            "Not enough valid labeled pixels. Check KMZ labels and raster."
        )

    all_x = np.moveaxis(features, 0, -1).reshape(-1, 8)
    labels = np.full(shape, -1, dtype=np.int8)
    labels[quarry] = 1
    labels[negatives] = 0
    labels = labels.reshape(-1)

    selected = labels >= 0
    X, y = all_x[selected], labels[selected]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predicted = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    try:
        auc = roc_auc_score(y_test, probabilities)
    except ValueError:
        auc = float("nan")

    metrics = {
        "Accuracy": accuracy_score(y_test, predicted),
        "Precision": precision_score(y_test, predicted, zero_division=0),
        "Recall": recall_score(y_test, predicted, zero_division=0),
        "F1 score": f1_score(y_test, predicted, zero_division=0),
        "ROC AUC": auc,
        "Test pixels": len(y_test),
        "Quarry training pixels": int(quarry.sum()),
        "Non-quarry training pixels": int(negatives.sum()),
        "Confusion matrix": confusion_matrix(
            y_test, predicted, labels=[0, 1]
        ).tolist(),
    }
    importance = dict(zip(FEATURE_NAMES, model.feature_importances_))
    return model, metrics, importance


def predict_raster(model, raster_path, threshold=0.60):
    """Return quarry probability, candidate mask, validity, and raster profile."""
    features, valid, profile, _, _ = read_features(raster_path)
    rows, cols = valid.shape
    X = np.moveaxis(features, 0, -1).reshape(-1, 8)
    valid_flat = valid.reshape(-1)

    probability = np.full(rows * cols, np.nan, dtype="float32")
    probability[valid_flat] = model.predict_proba(X[valid_flat])[:, 1]
    probability = probability.reshape(rows, cols)

    candidates = valid & (probability >= threshold)
    pixel_area = abs(
        profile["transform"].a * profile["transform"].e
        - profile["transform"].b * profile["transform"].d
    )
    area_ha = int(candidates.sum()) * pixel_area / 10000.0

    return probability, candidates, valid, profile, area_ha
