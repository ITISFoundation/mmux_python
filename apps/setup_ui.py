import streamlit as st
from mmux_utils.funs_evaluate import create_run_dir
from mmux_utils.funs_git import clone_repo, import_function_from_repo, check_repo_exists
import time
from pathlib import Path


def reset_clone_status():
    st.session_state.cloned = False


def main():
    ## TODO rather create this as a pure python script (no Streamlit)
    # and only then, move to making the GUI and see if it is powerful enough

    st.title("MetaModeling UX")
    st.header("Pipeline Setup")
    temp_dir = create_run_dir(Path("."), "setup")
    reset_clone_status()

    repo_url = st.text_input(
        "Model: ",
        placeholder="https://github.com/...",
        on_change=lambda: reset_clone_status(),
    )

    ## TODO try Copilot changes to make Clone button disappear once clicked; otherwise check what I saw in Overflow
    ## also probably much better program flow, with proper callbacks and so on (not so much, nested ifs)
    ## in any case, commit this version and be able to inspect diff solutions (in diff branches / commits?)
    if repo_url:
        valid_repo = check_repo_exists(repo_url)
        if valid_repo:
            if not st.session_state.cloned:
                if st.button("Clone"):
                    with st.spinner("Cloning..."):
                        status, message = clone_repo(repo_url, target_dir=temp_dir)
                        time.sleep(2)

                    if status == True:
                        st.success("✔️ Cloning succeeded!")
                        st.session_state.cloned = True  # Update session state
                    else:
                        st.error("❌ Cloning failed.")
                        st.error(message)
        else:
            st.warning("Please enter a valid GitHub repository URL!")

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
