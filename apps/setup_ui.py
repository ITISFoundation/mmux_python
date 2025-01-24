import streamlit as st
from mmux_utils.funs_evaluate import create_run_dir
from mmux_utils.funs_git import clone_repo, import_function_from_repo, check_repo_exists
import time
from pathlib import Path

temp_dir = create_run_dir(Path("."), "setup")


def reset_clone_status():
    st.session_state.cloned = False


if "cloned" not in st.session_state:
    reset_clone_status()


def clone(repo_url: str, temp_dir: Path):
    try:
        assert check_repo_exists(repo_url)

        with st.spinner("Cloning..."):
            status, message = clone_repo(repo_url, target_dir=temp_dir)
            time.sleep(2)

        if status == True:
            st.success("✔️ Cloning succeeded!")
            st.session_state.cloned = True  # Update session state
        else:
            st.error("❌ Cloning failed.")
            st.error(message)
    except:
        st.warning("Please enter a valid GitHub repository URL!")


def main():
    ## TODO rather create this as a pure python script (no Streamlit)
    # and only then, move to making the GUI and see if it is powerful enough

    st.title("MetaModeling UX")
    st.header("Pipeline Setup")

    repo_url = st.text_input(
        "Model: ",
        placeholder="https://github.com/...",
        on_change=lambda: reset_clone_status(),
    )

    st.button(
        "Clone",
        on_click=clone,
        args=(repo_url, temp_dir),
        disabled=st.session_state.cloned,
    )

    print("After button?", st.session_state.cloned)

    ## TODO try Copilot changes to make Clone button disappear once clicked; otherwise check what I saw in Overflow
    ## also probably much better program flow, with proper callbacks and so on (not so much, nested ifs)
    ## in any case, commit this version and be able to inspect diff solutions (in diff branches / commits?)

    ## TODO be able to extract input & output variables;
    # and generate the textboxes to input values for them

    # if st.button("Run"):  ## TODO only show when cloning is complete
    #     st.warning("Not implemented yet")

    #####################

    #         fun = import_function_from_repo(
    #             repo_path, module_name="evaluation.py", function_name="evaluator"
    #         )

    # # Run button
    # if st.button("Run"):
    #     if clone_repo and success:
    #         # Call backend logic
    #         # result = process_input(user_input)
    #         result = 0.0
    #         st.success(f"Result: {result}")
    #     else:
    #         st.warning("Please enter some input!")


if __name__ == "__main__":
    main()
