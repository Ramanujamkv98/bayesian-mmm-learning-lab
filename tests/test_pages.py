from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGE_FILES = [ROOT / "app.py", *sorted((ROOT / "pages").glob("*.py"))]


@pytest.mark.parametrize("page_path", PAGE_FILES, ids=lambda path: path.name)
def test_streamlit_page_starts_without_exception(page_path):
    app = AppTest.from_file(str(page_path), default_timeout=40)
    app.run()
    assert not app.exception
