"""Smoke tests for the Streamlit application.

app.py is the largest file in the project and had no coverage at all, because
a Streamlit script cannot simply be imported: running it *is* the test. The
framework ships AppTest for this, which executes the script and exposes its
widgets and output for inspection.

These are deliberately few and broad. Each run executes the whole model, so a
long suite here would dominate CI for little extra signal. What they establish
is that the application starts, that its controls actually change what they
claim to, and that no run raises.

Skipped when the visualisation extras are not installed, so the physics suite
still runs on a bare `pip install -e ".[dev]"`.
"""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit", reason="requires the viz extra")
pytest.importorskip("plotly", reason="requires the viz extra")

from pathlib import Path  # noqa: E402

from streamlit.testing.v1 import AppTest  # noqa: E402

TIMEOUT = 300

# AppTest resolves a relative path against the file that calls it, which would
# look for tests/app.py.
ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"


def run_app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT)
    at.run()
    return at


@pytest.fixture(scope="module")
def app():
    return run_app()


def test_application_starts_without_raising(app):
    assert not app.exception, app.exception


def test_the_two_top_level_views_exist(app):
    assert len(app.tabs) >= 2


def test_departure_board_and_charts_render(app):
    """The board is raw HTML, so it appears as markdown rather than a widget."""
    html = " ".join(str(m.value) for m in app.markdown)
    assert "board-row" in html
    assert "physics minimum" in html
    assert len(app.get("plotly_chart")) >= 10


def test_the_schedule_is_labelled_as_modelled(app):
    """The board must never read as a live feed. This is the one claim in the
    application that would be dishonest if it were dropped."""
    warnings = " ".join(str(w.value) for w in app.warning)
    assert "not a live feed" in warnings


def test_class_picker_changes_the_selected_class():
    """Uses session state and an explicit rerun, so it is worth checking that
    a click really does switch class rather than merely redraw."""
    at = run_app()
    assert at.session_state["cls_key"] == "E235-0"

    button = next(b for b in at.sidebar.button if b.key == "pick_E233-0")
    button.click().run()

    assert at.session_state["cls_key"] == "E233-0"
    assert not at.exception, at.exception


def test_time_of_day_changes_headway_and_crowding():
    """Scenario is the primary control: it must move several parameters at
    once, not just relabel the header."""
    at = run_app()
    scenario = at.sidebar.selectbox[0]
    assert "Daytime" in scenario.value

    metrics = {m.label: m.value for m in at.metric}
    daytime_headway = metrics["Headway"]

    scenario.select("Morning peak").run()
    assert not at.exception, at.exception

    peak = {m.label: m.value for m in at.metric}
    assert peak["Headway"] != daytime_headway
    assert peak["Crowding"] == "125%"


def test_no_metric_label_appears_twice(app):
    """Merging three views into one page previously left circuit time and
    crowding stated twice."""
    labels = [m.label for m in app.metric]
    assert len(labels) == len(set(labels)), (
        f"duplicated metric labels: "
        f"{[x for x in labels if labels.count(x) > 1]}"
    )


def test_source_does_not_use_removed_streamlit_parameters():
    """use_container_width was deprecated with a removal date that has passed.
    Cheap static check, so it stays honest without another app run."""
    source = APP.read_text(encoding="utf-8")
    assert "use_container_width" not in source
