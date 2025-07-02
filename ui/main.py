import streamlit as st
import sys
import config
import pandas as pd
from PIL import Image 

def main():
    imagesize = config.IMAGE_SIZE
    image_path = config.EXAMPLE_IMAGE
    st.title("Imagination in Translation")


    if "show_image" not in st.session_state:
        st.session_state.show_image = True
    
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    # Load and display an image
     

    image = Image.open(image_path) 
    st.image(image, caption="Sample Image")

    st.write("Please describe what you see.")
    st.session_state.user_input = st.text_area("Your description:", value=st.session_state.user_input, height=imagesize,key="Description")
    if st.button("Submit"):
            text = st.session_state.user_input

    
if __name__ == "__main__":
    main()
