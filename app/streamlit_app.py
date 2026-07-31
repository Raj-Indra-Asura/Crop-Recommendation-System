"""The Streamlit demo UI for the crop recommendation model (Week 10).

Run it from the repository root::

    streamlit run app/streamlit_app.py

It calls :func:`src.pipelines.predict_pipeline.predict` **directly, in the same
Python process**. It does *not* send an HTTP request to ``api.main``, and the
FastAPI server does not have to be running for this app to work. That decision,
and its consequences, are argued in
``docs/curriculum/week10/learning_notes.md`` §5.

This is a demo, not a production frontend: Streamlit re-runs this whole script
top to bottom on every widget interaction, holds session state in server
memory, and has no authentication or rate limiting. It exists so a human can
see the model answer, and it is not what real traffic should hit.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.pipelines.predict_pipeline import EXAMPLE_INPUT, load_pipeline, predict, predict_proba

#: Label, unit, bounds and default for each of the seven inputs. Bounds match
#: `api.schemas.CropFeatures`, so the form and the API agree on what is valid;
#: the defaults are the example input used everywhere else in the project.
FEATURE_FIELDS: list[dict[str, Any]] = [
    {"name": "N", "label": "Nitrogen (N)", "unit": "kg/ha", "min": 0.0, "max": 500.0, "step": 1.0},
    {
        "name": "P",
        "label": "Phosphorus (P)",
        "unit": "kg/ha",
        "min": 0.0,
        "max": 500.0,
        "step": 1.0,
    },
    {"name": "K", "label": "Potassium (K)", "unit": "kg/ha", "min": 0.0, "max": 500.0, "step": 1.0},
    {
        "name": "temperature",
        "label": "Temperature",
        "unit": "°C",
        "min": -20.0,
        "max": 60.0,
        "step": 0.1,
    },
    {"name": "humidity", "label": "Humidity", "unit": "%", "min": 0.0, "max": 100.0, "step": 0.1},
    {"name": "ph", "label": "Soil pH", "unit": "0-14", "min": 0.0, "max": 14.0, "step": 0.1},
    {
        "name": "rainfall",
        "label": "Rainfall",
        "unit": "mm",
        "min": 0.0,
        "max": 1000.0,
        "step": 1.0,
    },
]

#: How many crops the result panel lists.
TOP_K = 3


@st.cache_resource(show_spinner="Loading the trained model…")
def get_pipeline():
    """Load the fitted pipeline once per Streamlit server process.

    Streamlit re-runs this script on every interaction, so an uncached
    ``load_pipeline()`` would re-read — or, on a clean clone, re-train — the
    model every time a slider moved. ``st.cache_resource`` keeps one object for
    the life of the server, which is the same "load once, serve many" rule the
    API follows in :mod:`api.main`.

    Returns:
        The fitted :class:`~sklearn.pipeline.Pipeline`.
    """
    return load_pipeline()


def render() -> None:
    """Draw the page: a form for the seven features, and the prediction."""
    st.set_page_config(page_title="Crop Recommendation", page_icon="🌱")
    st.title("🌱 Crop Recommendation")
    st.caption(
        "Enter the soil and weather measurements of a plot, and the Week 9 model "
        "recommends one of 22 crops. Demo UI — not a production frontend."
    )

    with st.form("features"):
        columns = st.columns(2)
        values: dict[str, float] = {}
        for index, field in enumerate(FEATURE_FIELDS):
            with columns[index % 2]:
                values[field["name"]] = st.number_input(
                    f"{field['label']} ({field['unit']})",
                    min_value=field["min"],
                    max_value=field["max"],
                    value=float(EXAMPLE_INPUT[field["name"]]),
                    step=field["step"],
                )
        submitted = st.form_submit_button("Recommend a crop")

    if not submitted:
        st.info("The form is pre-filled with the example from the docs. Press the button.")
        return

    try:
        pipeline = get_pipeline()
        label = predict(values, pipeline=pipeline)
        ranked = predict_proba(values, pipeline=pipeline, top_k=TOP_K)
    except Exception as error:  # noqa: BLE001 - a UI reports failures, it does not crash
        st.error(f"Could not produce a recommendation: {error}")
        return

    st.success(f"Recommended crop: **{label}**")
    st.metric("Confidence", f"{ranked.get(label, 0.0):.1%}")
    st.subheader(f"Top {len(ranked)} candidates")
    st.bar_chart({"probability": ranked})
    st.caption(
        "A close second place means the measurements sit between two crops — the "
        "`rice` / `jute` pair from Week 8 is the usual culprit."
    )


# Streamlit executes this module as a script, so the page is drawn on import
# *when run through `streamlit run`*. Guarding on `st.runtime` keeps a plain
# `python -c "import app.streamlit_app"` (the Week 10 syntax check) silent.
if st.runtime.exists():
    render()
