import pytest
from streamlit.testing.v1 import AppTest
from utils.funs_git import get_mmux_top_directory

UI = get_mmux_top_directory() / "apps" / "setup_ui.py"


def test_createAppTest():
    _ = AppTest.from_file(UI)


def test_runAppTest():
    at = AppTest.from_file(UI)
    at.run()


def test_app_initialization():
    at = AppTest.from_file(UI)
    at.run()
    assert not at.exception
    assert at.header[0].value == "Pipeline Setup"


@pytest.mark.skip(reason="The cloned state variable doesnt seem to keep state?!?!")
def test_clone_repo():
    at = AppTest.from_file(UI)
    at.run()
    assert not at.exception

    at.text_input[0].input(
        "https://github.com/ITISFoundation/optistim-pulse-evaluation"
    )
    at.button[0].click().run()
    # FIXME
    # Getting inside button callback!
    # Unknown error when cloning the repository: fatal: The empty string is not a valid path

    assert at.session_state.cloned
    print("Done")
