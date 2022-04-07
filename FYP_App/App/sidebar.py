import streamlit as st
import re
import datetime

def show():
    """Shows the sidebar components for the template and returns user input as dict."""
    input = {}

    with st.sidebar:

    #------------------------------------------------------------------------

        st.write("## Problem Set up")
        input["DateType"] = st.selectbox(
                            "Choice of date: random or Specified?", 
                            ("Specified", "Random"),
            )
        with st.form(key='my_form'):
           
            if input["DateType"]=="Specified":

                input["StartEpoch"] = st.date_input(
                            "Starting Epoch",
                            datetime.date(2001, 7, 6), 
                            min_value=datetime.date(2000, 1, 1),
                            max_value=datetime.date(2050, 12, 31),
                )

                input["EndEpoch"] = st.date_input(
                            "Ending Epoch",
                            datetime.date(2002, 7, 6), 
                            min_value=datetime.date(2000, 1, 1),
                            max_value=datetime.date(2050, 12, 31),
                )

            shortway = st.selectbox(
                                "Solve for short way theta < 180  or long way theta > 180 ?",
                                ("Short", "Long"),
            )
            

            input["short_way"] = True if shortway=="Short" else False

            plot = st.selectbox(
                            "Plot solution?",
                            ("Yes", "No"),
            )
            input["plot"] = True if plot=="Yes" else False

        
            input["GetLambert"] = st.form_submit_button("Generate Lambert Problem")

            # if input["DateType"]=="Random" :
            #     st.write("Please specify starting date limits and Time of Flight Limits")
        st.markdown('#')

        with st.form(key='my_form2'):
            input['num_epochs'] = st.number_input("Enter Number of epochs for model training", 
                                                  min_value=1, max_value=10000, value=1001)

            input['lr'] = st.number_input("Enter learning rate for model training", 
                                                  min_value=1e-5, max_value=2.0, value=2e-2, step=1e-2, format='%.2e')
            input["print_interval"] = st.number_input("Enter Print Interval for display of loss during training", 
                                                  min_value=1, max_value=10000, value=100)
            input["run"] = st.form_submit_button("Fit Model")

        return input

