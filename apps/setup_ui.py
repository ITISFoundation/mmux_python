import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from utils.funs_evaluate import create_run_dir
from utils.funs_git import clone_repo, get_attr_from_repo, check_repo_exists
from typing import Callable
from pathlib import Path
from apps.styles import header_style


def reset_clone_status():
    st.session_state.cloned = False
    temp_dir = create_run_dir(Path("."), "setup")
    st.session_state.run_dir = temp_dir


def clone(repo_url: str, mssg_container: DeltaGenerator):
    print("Getting inside button callback!")
    try:
        assert check_repo_exists(repo_url)
    except Exception:
        mssg_container.warning("Please enter a valid GitHub repository URL!")

    with mssg_container:
        with st.spinner("Cloning..."):
            status, message = clone_repo(
                repo_url,
                target_dir=st.session_state.run_dir,
            )

    with mssg_container.empty():
        if status:
            ## NB: center-alignment doesnt seem to be natively possible in Streamlit
            ## the only option is to use HTML in Markdown and set style='text-align: center; ..
            st.success("✔️ Cloning succeeded!")
            st.session_state.cloned = True  # Update session state
            print(message)
        else:
            st.error("❌ Cloning failed.")
            print(message)
            st.error(message)


#### Define page style & setup   ###################################
####################################################################
# Set page configuration
st.set_page_config(page_title="MetaModeling UX", layout="wide")
st.markdown(header_style, unsafe_allow_html=True)  # Purple header
st.markdown('<div class="purple-header">MetaModeling UX</div>', unsafe_allow_html=True)
if "cloned" not in st.session_state:
    reset_clone_status()
st.header("Pipeline Setup")
####################################################################

#

#### Define page layout          ###################################
#### PS this decouples layout & functionality!
####################################################################
(left_col, right_half) = st.columns([1, 1])  # Split into columns
label_col, input_col, button_placeholder = left_col.columns(
    [0.5, 3, 1],  # Create a row with columns for label, input & button
    vertical_alignment="center",
)
####################################################################

#### MAIN APPLICATION ##############################################
####################################################################
## First element
label_col.text("Model: ")
repo_url = input_col.text_input(
    label="Model Input",
    label_visibility="collapsed",
    placeholder="https://github.com/...",
    on_change=lambda: reset_clone_status(),
)
lock = st.session_state.cloned and bool(repo_url)
button_placeholder.button(
    label="Clone",
    on_click=clone,
    args=(repo_url, left_col),
    disabled=lock,
)

if st.session_state.cloned:
    left_col.write("Trying to extract 'model' function...")
    ## TODO be able to extract input & output variables;
    # and generate the textboxes to input values for them
    model_function: Callable = get_attr_from_repo(
        st.session_state.run_dir,
        "main.py",
        "model",
    )

    left_col.write("'model' function extracted successfully!")
    print(model_function.__annotations__)
