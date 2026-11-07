import streamlit as st

st.set_page_config(page_title="DIP Lab 3", layout="wide")

st.title("Digital Image Processing - Lab 3")
st.write("Choose an experiment from the sidebar on the left:")

st.markdown(
    """
        - **Fourier Transform**: DFT of the cameraman image, low-pass / high-pass filtering in the frequency domain, and back to the spatial domain.
        - **Image Restoration**: noise removal on the Lenna image (Gaussian noise + adaptive median filter, salt & pepper noise + mean / median / max / min filters).
    """
)

st.info(
    "Each page has its own controls in the sidebar. "
    "You can upload your own image, or leave it empty to use the default one."
)