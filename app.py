from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
from pyproj import Transformer
from rasterio.io import MemoryFile
from scipy import ndimage

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image as PDFImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# This app expects quarry_model.py to define:
#   train_model(raster_path, labels_dir) -> (model, metrics, importance)
#   predict_raster(model, raster_path, threshold) ->
#       (probability, candidates, valid, profile, area_ha)
from quarry_model import train_model, predict_raster


# ---------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

BEFORE_RASTER = DATA_DIR / "BhediKharka_Before_20200124_SCL.tif"
AFTER_RASTER = DATA_DIR / "BhediKharka_After_20211222_SCL.tif"

LABEL_FILES = [
    "Known_Quarry_2021.kmz",
    "NonQuarry_Water_2021.kmz",
    "NonQuarry_RiverSediment_2021.kmz",
    "NonQuarry_Roads_2021.kmz",
    "NonQuarry_Agriculture_2021.kmz",
]

st.set_page_config(
    page_title="The Change Sentinel",
    page_icon="🛰️",
    layout="wide",
)

st.title("🛰️ The Change Sentinel")
st.caption(
    "Satellite-based screening for quarry-like surface change. "
    "Detections are preliminary leads, not findings of illegality."
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def raster_bytes(array, profile, dtype=None, nodata=None):
    """Create a single-band GeoTIFF in memory for download."""
    out_profile = profile.copy()
    out_profile.update(
        driver="GTiff",
        count=1,
        compress="deflate",
        dtype=str(dtype or array.dtype),
    )
    if nodata is not None:
        out_profile["nodata"] = nodata

    # GeoTIFF files must contain finite numeric values where possible.
    clean = np.asarray(array).copy()
    if np.issubdtype(clean.dtype, np.floating):
        clean = np.nan_to_num(clean, nan=float(nodata if nodata is not None else -9999))
    with MemoryFile() as memfile:
        with memfile.open(**out_profile) as dst:
            dst.write(clean.astype(out_profile["dtype"]), 1)
        return memfile.read()


def make_candidate_map(probability, selected_mask=None, valid_mask=None, title=None):
    """Render a probability map to PNG bytes."""
    fig, ax = plt.subplots(figsize=(8, 5))
    shown = np.asarray(probability, dtype="float32")
    if valid_mask is not None:
        shown = np.where(valid_mask, shown, np.nan)

    im = ax.imshow(shown, cmap="magma", vmin=0, vmax=1)
    if selected_mask is not None and np.any(selected_mask):
        ax.contour(
            selected_mask.astype("uint8"),
            levels=[0.5],
            colors="cyan",
            linewidths=1.2,
        )
    ax.set_title(title or "Quarry-like screening score")
    ax.set_xlabel("Raster column")
    ax.set_ylabel("Raster row")
    fig.colorbar(im, ax=ax, label="Model screening score")
    fig.tight_layout()

    output = BytesIO()
    fig.savefig(output, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output.getvalue()


def safe_paragraph(value):
    """Escape user-entered text before inserting it into a ReportLab Paragraph."""
    return escape(str(value or "Not provided")).replace("\n", "<br/>")


def make_case_pdf(case_id, details, map_png):
    """Build a preliminary case dossier PDF in memory."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )
    styles = getSampleStyleSheet()

    story = [
        Paragraph("THE CHANGE SENTINEL", styles["Title"]),
        Paragraph("Preliminary Satellite Evidence Dossier", styles["Heading2"]),
        Spacer(1, 8),
        Paragraph(f"<b>Case reference:</b> {safe_paragraph(case_id)}", styles["Normal"]),
        Paragraph(
            "<b>Classification:</b> Suspected quarry-like surface — verification required",
            styles["Normal"],
        ),
        Paragraph(
            f"<b>Generated (UTC):</b> {safe_paragraph(details['generated'])}",
            styles["Normal"],
        ),
        Spacer(1, 10),
        Paragraph("1. Candidate summary", styles["Heading2"]),
    ]

    rows = [
        ["Field", "Value"],
        ["Site description", safe_paragraph(details["site"])],
        ["District / region", safe_paragraph(details["region"])],
        ["Receiving authority", safe_paragraph(details["authority"])],
        ["Estimated candidate area", f"{details['area_ha']:.2f} hectares"],
        ["Candidate pixels", f"{details['pixels']:,}"],
        ["Mean screening score", f"{details['mean_score']:.3f}"],
        ["Maximum screening score", f"{details['max_score']:.3f}"],
        ["Screening threshold", f"{details['threshold']:.2f}"],
        ["Latitude", f"{details['latitude']:.6f}"],
        ["Longitude", f"{details['longitude']:.6f}"],
        ["Centroid in raster CRS", f"{details['x_map']:.2f}, {details['y_map']:.2f}"],
        ["Raster CRS", safe_paragraph(details["crs"])],
        ["Before imagery date", "2020-01-24 (source filename)"],
        ["After imagery date", "2021-12-22 (source filename)"],
    ]
    table = Table(rows, colWidths=[5.0 * cm, 11.0 * cm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce6f1")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))

    story.extend([
        table,
        Spacer(1, 12),
        Paragraph("2. Candidate map", styles["Heading2"]),
        PDFImage(BytesIO(map_png), width=16 * cm, height=10 * cm),
        Spacer(1, 8),
        Paragraph("3. Reviewer observations", styles["Heading2"]),
        Paragraph(safe_paragraph(details["observations"]), styles["BodyText"]),
        Spacer(1, 8),
        Paragraph("4. Verification checklist", styles["Heading2"]),
        Paragraph(
            "• Verify the site coordinates and boundary through field inspection.<br/>"
            "• Check applicable mining permits or lease records with the competent authority.<br/>"
            "• Check environmental clearances and other approvals where applicable.<br/>"
            "• Compare dated satellite imagery with field observations and official records.<br/>"
            "• Preserve source imagery, acquisition metadata, and field photographs.",
            styles["BodyText"],
        ),
        Spacer(1, 8),
        Paragraph("5. Draft request for official verification", styles["Heading2"]),
        Paragraph("To: " + safe_paragraph(details["authority"]), styles["BodyText"]),
        Spacer(1, 5),
        Paragraph(
            "<b>Subject: Request to verify suspected quarry-like land-surface activity</b>",
            styles["BodyText"],
        ),
        Spacer(1, 5),
        Paragraph(
            "Respected Sir/Madam,<br/><br/>"
            "I request that the competent authority review the location identified in the "
            "attached preliminary satellite-screening dossier and determine whether an "
            "inspection is warranted. Automated screening identified a quarry-like surface "
            "pattern at the coordinates and estimated area listed in this report. This is a "
            "preliminary lead only; it does not establish that extraction occurred or that "
            "any activity is unlawful.<br/><br/>"
            "Kindly verify the site boundary, relevant permits or lease records, and applicable "
            "environmental approvals, and take any action considered appropriate after "
            "independent verification. Supporting map and screening details are attached."
            "<br/><br/>Yours faithfully,<br/>"
            "[Submitter name and contact details to be completed]",
            styles["BodyText"],
        ),
        Spacer(1, 8),
        Paragraph("Important limitations", styles["Heading2"]),
        Paragraph(
            "This report is machine-generated screening evidence, not proof of illegal "
            "quarrying, an official survey, or a legal finding. Model scores are not the "
            "probability that a law was broken. Coordinate accuracy depends on the source "
            "raster and georeferencing. A human reviewer must validate the site and evidence "
            "before submission.",
            styles["BodyText"],
        ),
    ])

    doc.build(story)
    return buffer.getvalue()


def required_files_missing():
    expected = [BEFORE_RASTER, AFTER_RASTER] + [DATA_DIR / name for name in LABEL_FILES]
    return [path for path in expected if not path.is_file()]


@st.cache_resource(show_spinner="Training the screening model…")
def get_trained_model(after_raster, labels_dir):
    # Keep paths as arguments so Streamlit cache keys are stable.
    return train_model(after_raster, labels_dir)


# ---------------------------------------------------------------------
# Validate data and prepare model
# ---------------------------------------------------------------------
missing = required_files_missing()
if missing:
    st.error(
        "Required files are missing. Place the two rasters and all five KMZ label files "
        "inside the `data` folder. Filenames must match exactly:"
    )
    for path in missing:
        st.code(str(path))
    st.stop()

try:
    model, metrics, importance = get_trained_model(str(AFTER_RASTER), str(DATA_DIR))
except Exception as exc:
    st.error(
        "The model could not be trained. Check `quarry_model.py`, the raster/KMZ files, "
        "and the installed packages."
    )
    st.exception(exc)
    st.stop()

threshold = 0.60  # Fixed screening threshold; slider removed.

try:
    probability, candidates, valid, profile, _ = predict_raster(
        model, str(AFTER_RASTER), threshold
    )
except Exception as exc:
    st.error("Could not run prediction on the after-image raster.")
    st.exception(exc)
    st.stop()

# Standardize candidate/valid arrays to boolean for display and component labeling.
candidates = np.asarray(candidates).astype(bool)
valid = np.asarray(valid).astype(bool)
probability = np.asarray(probability, dtype="float32")

tabs = st.tabs([
    "Probability map",
    "Upload & predict",
    "Before / after",
    "Model performance",
    "Generate case",
])


# ---------------------------------------------------------------------
# Tab 1: Main map and downloads
# ---------------------------------------------------------------------
with tabs[0]:
    st.subheader("Quarry-like surface screening")

    transform = profile["transform"]
    pixel_area_m2 = abs(transform.a * transform.e - transform.b * transform.d)
    candidate_area_ha = int(candidates.sum()) * pixel_area_m2 / 10000

    c1, c2, c3 = st.columns(3)
    c1.metric("Candidate pixels", f"{int(candidates.sum()):,}")
    c2.metric("Estimated candidate area", f"{candidate_area_ha:.2f} ha")
    c3.metric("Screening threshold", f"{threshold:.2f}")

    # Display a compact, centered probability map (no resize buttons).
    map_png = make_candidate_map(
        probability,
        valid_mask=valid,
        title="Quarry-like screening score — 2021-12-22",
    )

    _, image_col, _ = st.columns([1, 3, 1])
    with image_col:
        st.image(
            map_png,
            caption="Quarry-like screening score — 22 Dec 2021",
            width=450,
        )

    st.download_button(
        "Download probability GeoTIFF",
        data=raster_bytes(probability, profile, dtype="float32", nodata=-9999),
        file_name="quarry_probability_20211222.tif",
        mime="image/tiff",
    )
    st.download_button(
        "Download candidate mask GeoTIFF",
        data=raster_bytes(candidates.astype("uint8"), profile, dtype="uint8", nodata=0),
        file_name="quarry_candidates_20211222.tif",
        mime="image/tiff",
    )


# ---------------------------------------------------------------------
# Tab 2: Upload validation / preview
# ---------------------------------------------------------------------
with tabs[1]:
    st.subheader("Upload and inspect a compatible raster")
    st.write(
        "Expected format: seven-band GeoTIFF with bands in this order: "
        "B2, B3, B4, B8, B11, B12, SCL."
    )
    uploaded = st.file_uploader(
        "Choose a GeoTIFF",
        type=["tif", "tiff"],
        key="uploaded_raster",
    )

    if uploaded is not None:
        try:
            with MemoryFile(uploaded.getvalue()) as memfile:
                with memfile.open() as src:
                    upload_profile = src.profile.copy()
                    upload_count = src.count
                    upload_width = src.width
                    upload_height = src.height
                    upload_crs = src.crs

            if upload_count != 7:
                st.error(f"This file has {upload_count} bands; the expected number is 7.")
            else:
                st.success("The file has seven bands and can be read as a compatible input.")
                st.write(f"Dimensions: {upload_width} × {upload_height} pixels")
                st.write(f"CRS: {upload_crs}")
                st.info(
                    "This tab currently validates the uploaded raster. To predict on arbitrary uploaded "
                    "rasters, `quarry_model.py` must expose a prediction function that accepts an in-memory "
                    "array/profile rather than only a file path."
                )
        except Exception as exc:
            st.error(f"Could not read this GeoTIFF: {exc}")


# ---------------------------------------------------------------------
# Tab 3: Before/after NDVI comparison
# ---------------------------------------------------------------------
with tabs[2]:
    st.subheader("Before and after vegetation-index comparison")
    try:
        with rasterio.open(BEFORE_RASTER) as src:
            before = src.read()
        with rasterio.open(AFTER_RASTER) as src:
            after = src.read()

        # Band indexes are zero-based: B4 is index 2; B8 is index 3.
        before_red = before[2].astype("float32")
        before_nir = before[3].astype("float32")
        after_red = after[2].astype("float32")
        after_nir = after[3].astype("float32")

        before_den = before_nir + before_red
        after_den = after_nir + after_red

        before_ndvi = np.divide(
            before_nir - before_red,
            before_den,
            out=np.full(before_den.shape, np.nan, dtype="float32"),
            where=before_den != 0,
        )
        after_ndvi = np.divide(
            after_nir - after_red,
            after_den,
            out=np.full(after_den.shape, np.nan, dtype="float32"),
            where=after_den != 0,
        )

        left, right = st.columns(2)
        with left:
            fig, ax = plt.subplots(figsize=(3.2, 2.4))
            ax.imshow(before_ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
            ax.set_title("NDVI before — 2020-01-24", fontsize=9)
            ax.axis("off")
            fig.tight_layout(pad=0.2)
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)
        with right:
            fig, ax = plt.subplots(figsize=(3.2, 2.4))
            ax.imshow(after_ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
            ax.set_title("NDVI after — 2021-12-22", fontsize=9)
            ax.axis("off")
            fig.tight_layout(pad=0.2)
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

        st.caption(
            "NDVI is one change indicator only. Season, atmospheric conditions, and land management "
            "can also cause differences."
        )
    except Exception as exc:
        st.warning(f"Could not create the before/after comparison: {exc}")


# ---------------------------------------------------------------------
# Tab 4: Model performance
# ---------------------------------------------------------------------
with tabs[3]:
    st.subheader("Model performance and feature importance")
    st.caption(
        "Random pixel train/test splits can overstate performance when nearby pixels are similar. "
        "Test on separate locations or spatial blocks before operational use."
    )

    if isinstance(metrics, dict):
        st.dataframe(pd.DataFrame([metrics]), use_container_width=True)
    elif isinstance(metrics, pd.DataFrame):
        st.dataframe(metrics, use_container_width=True)
    else:
        st.write(metrics)

    if isinstance(importance, dict):
        importance_df = pd.DataFrame({
            "Feature": list(importance.keys()),
            "Importance": list(importance.values()),
        }).sort_values("Importance", ascending=False)
        st.dataframe(importance_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(
            importance_df["Feature"].iloc[::-1],
            importance_df["Importance"].iloc[::-1],
        )
        ax.set_title("Random Forest feature importance")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    elif isinstance(importance, pd.DataFrame):
        st.dataframe(importance, use_container_width=True)
    else:
        st.write(importance)


# ---------------------------------------------------------------------
# Tab 5: Case dossier generator
# ---------------------------------------------------------------------
with tabs[4]:
    st.subheader("Generate a preliminary case dossier")
    st.write(
        "Select a connected region from the model's candidate mask. The app creates a downloadable "
        "PDF with coordinates, area estimate, map, and a neutral request for official verification."
    )
    st.warning(
        "This does not automatically file a case. A person must verify the evidence, identify the "
        "competent authority, complete the draft, and submit it through the appropriate official process."
    )

    labels, number_of_regions = ndimage.label(
        candidates,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    region_sizes = np.bincount(labels.ravel())
    region_ids = [
        idx for idx in range(1, number_of_regions + 1)
        if region_sizes[idx] >= 5
    ]
    region_ids.sort(key=lambda idx: region_sizes[idx], reverse=True)

    if not region_ids:
        st.info(
            "No connected candidate region with at least five pixels was found at this threshold. "
            "The screening threshold is fixed at 0.60 in this version."
        )
    else:
        transform = profile["transform"]
        pixel_area_m2 = abs(transform.a * transform.e - transform.b * transform.d)
        pixel_area_ha = pixel_area_m2 / 10000

        def format_region(idx):
            return (
                f"Candidate {idx} — {region_sizes[idx]:,} pixels, "
                f"{region_sizes[idx] * pixel_area_ha:.2f} ha"
            )

        selected_id = st.selectbox(
            "Choose a candidate region",
            options=region_ids[:300],
            format_func=format_region,
        )
        selected_mask = labels == selected_id
        pixel_coords = np.argwhere(selected_mask)
        pixel_count = int(pixel_coords.shape[0])
        area_ha = pixel_count * pixel_area_ha
        mean_score = float(np.nanmean(probability[selected_mask]))
        max_score = float(np.nanmax(probability[selected_mask]))

        center_row = float(pixel_coords[:, 0].mean())
        center_col = float(pixel_coords[:, 1].mean())
        x_map, y_map = rasterio.transform.xy(
            transform,
            center_row,
            center_col,
            offset="center",
        )

        if profile.get("crs") is not None:
            transformer = Transformer.from_crs(
                profile["crs"], "EPSG:4326", always_xy=True
            )
            longitude, latitude = transformer.transform(x_map, y_map)
        else:
            longitude, latitude = float("nan"), float("nan")

        left, right = st.columns([1, 1])
        with left:
            st.metric("Estimated candidate area", f"{area_ha:.2f} ha")
            st.metric("Mean screening score", f"{mean_score:.3f}")
            st.write(f"**Approximate latitude:** {latitude:.6f}")
            st.write(f"**Approximate longitude:** {longitude:.6f}")
            st.write(f"**Raster CRS:** {profile.get('crs')}")
            st.caption(
                "The coordinates represent an approximate cluster centre, not a surveyed legal boundary."
            )

        with right:
            map_png = make_candidate_map(
                probability,
                selected_mask=selected_mask,
                valid_mask=valid,
                title=f"Candidate {selected_id} — selected region in cyan",
            )
            st.image(
                map_png,
                caption="Selected candidate region outlined in cyan",
                use_container_width=True,
            )

        with st.form("case_details_form"):
            site = st.text_input(
                "Site description (optional)",
                placeholder="Nearby village, landmark, or locality",
            )
            region = st.text_input("District / region (optional)")
            authority = st.text_input(
                "Competent authority (optional)",
                placeholder="Relevant district mining/geology or environmental authority",
            )
            observations = st.text_area(
                "Reviewer observations (optional)",
                placeholder="Enter factual observations only. Do not include unverified allegations.",
            )
            reviewed = st.checkbox(
                "I understand this is preliminary screening evidence and requires independent verification."
            )
            generate = st.form_submit_button(
                "Generate PDF dossier",
                disabled=not reviewed,
            )

        if generate:
            case_id = "CS-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            details = {
                "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
                "site": site,
                "region": region,
                "authority": authority or (
                    "The competent district mining/geology or environmental authority"
                ),
                "observations": observations,
                "area_ha": area_ha,
                "pixels": pixel_count,
                "mean_score": mean_score,
                "max_score": max_score,
                "threshold": threshold,
                "latitude": latitude,
                "longitude": longitude,
                "x_map": x_map,
                "y_map": y_map,
                "crs": profile.get("crs"),
            }
            st.session_state["case_pdf"] = make_case_pdf(case_id, details, map_png)
            st.session_state["case_id"] = case_id
            st.session_state["case_filename"] = f"{case_id}_preliminary_dossier.pdf"

        if st.session_state.get("case_pdf"):
            st.success(f"Dossier prepared: {st.session_state['case_id']}")
            st.download_button(
                "Download case dossier (PDF)",
                data=st.session_state["case_pdf"],
                file_name=st.session_state.get(
                    "case_filename", "preliminary_case_dossier.pdf"
                ),
                mime="application/pdf",
            )

st.sidebar.markdown("---")
st.sidebar.caption("The Change Sentinel • Preliminary screening only")
