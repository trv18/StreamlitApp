import streamlit as st
from jinja2 import Environment, FileSystemLoader
import uuid
from github import Github
from dotenv import load_dotenv
import os
import collections


import utils

# Choose your own Emoji
EMOJI_URL = "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/twitter/259/mage_1f9d9.png"

# Set page title and favicon.
st.set_page_config(
    page_title="FYP 2021-2022", page_icon=EMOJI_URL,
)


# Display header.
st.markdown("<br>", unsafe_allow_html=True)
st.image(EMOJI_URL, width=80)
# create

st.markdown(f'<p style="text-align:left;color:#ffffff ;font-size:20px;border-radius:0%;"> \
                        <b> Final Year Project <br>\
                        Imperial College London </b> <br> <br> \
                        Using Physics Informed Neural Networks to solve Lamberts Problem <br>\
                        1. Set Up Problem <br> \
                        2. Train PINN  <br> \
                        3. Plot Solution  <br> \
                        <hr> \
                        </p>', unsafe_allow_html=True)
#------------------------------------------------------------------------

with st.sidebar:
    st.info(
        "🎈 **NEW Code:** This app is still in development"
    )
    # st.error(
    #     "Found a bug? [Report it](https://github.com/jrieke/traingenerator/issues) 🐛"
    # )


#------------------------------------------------------------------------

# Show template-specific sidebar components (based on sidebar.py in the template dir).
template_sidebar = utils.import_from_file("sidebar", "./app/sidebar.py")

inputs = template_sidebar.show()
PINN = utils.import_from_file("PINN", "./app/LambertEq_training.py")


l = PINN.LambertEq()
if 'l' in st.session_state:
    if inputs["GetLambert"]:
        st.session_state['l'].Get_Lambert(new=True, print=True, shortway=inputs['short_way'], inputs=inputs)
    else:
        st.session_state['l'].Get_Lambert(new=False, print=True, shortway=inputs['short_way'], inputs=inputs)

if 'l' not in st.session_state:
    if inputs["GetLambert"]:
        st.session_state['l'] = l
        st.session_state['l'].Get_Lambert(shortway=inputs['short_way'], inputs=inputs)
        


if inputs["run"]:
    st.session_state['l'].TrainModel(num_epochs=inputs['num_epochs'], lr=inputs['lr'])
    st.session_state['l'].Get_Error()
