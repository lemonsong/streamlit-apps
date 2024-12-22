import streamlit as st
st.set_page_config(
        page_title="Market Analysis",
        page_icon="📈",
        layout="centered",
        initial_sidebar_state="expanded",
        # menu_items = {
        #     "Get Help": "https://streamlit.io",
        #     "Report a bug": "https://github.com",
        #     "About": "About my application **Hello World!**"
        # }
    )
@st.cache_data
def main():
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """

    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


if __name__ == "__main__":
    main()