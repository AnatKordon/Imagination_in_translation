import streamlit as st
import sys
import config
import pandas as pd
from PIL import Image 

def main():


    image_path = config.EXAMPLE_IMAGE
    st.title("Imagination in Translation")


    if "show_image" not in st.session_state:
        st.session_state.show_image = True

    if st.button("End of remembering"):
        st.session_state.show_image = not st.session_state.show_image
    # Load and display an image
    image = Image.open(image_path)  
    if st.session_state.show_image:
        st.image(image, caption="Sample Image")
    
if __name__ == "__main__":
    main()