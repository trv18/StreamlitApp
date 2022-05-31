import streamlit as st
import re
import datetime
import numpy as np
from ast import literal_eval


def show():
    """Shows the sidebar components for the template and returns user input as dict."""
    input = {}

    with st.sidebar:
        ### Set up Creation of Lmabert Problem ###
        st.write("## Problem Set up")
        input["DateType"] = st.selectbox(
                                "Choice of date: random or Specified?", 
                                ("Specified", "Random"),
                )

        ### Create Random or Specified Problem only when requested ###
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


    #------------------------------------------------------------------------
        ### Set up Solver to solve Lambert Problem ###
        st.write("## Training Set up")
        approach = st.selectbox(
                                        "Solution Approach: How would you like to solve the problem?",
                                        ("Theory of Functional Connections", "Lambert's Equation")
            )
        input["Model"] = 'TFC' if approach == "Theory of Functional Connections" else 'LambertEq'
        
        ### if TFC use appropriate config options ###
        if input['Model']=='TFC' :
            Sweep = st.selectbox(
                    "Do You Wish to perform multiple training runs to Quantify performance?",
                    ('No','Yes'),
                    )
            Sweep = True if Sweep=="Yes" else False
        
            ### Determine if sweep is to be used ###
            if not Sweep:
                with st.form(key='my_form2'):
                    input['num_runs'] = 1
                    input['Points'] = [st.number_input("Enter Number of traininig points for model training", 
                                                        min_value=1, max_value=1000, value=50, step=10)]
                                                        
                    input['Polynomial'] = [st.selectbox("Orthogonal Polynomials to be used", 
                                                        ("LeP", "CP", "FS"))]

                    input['Order'] = [st.number_input("Enter Max order of Polynomials to be used for model training", 
                                                        min_value=1, max_value=1000, value=50, step=10)]

                    J2 = st.selectbox("Incorporate J2 Dynamics??",
                                (" No", "Yes"),
                                )
                    input["Include_J2"] = True if J2=="Yes" else False

                    input["run"] = st.form_submit_button("Fit Model")


            else:
                ### If Sweeping enter all config combos ###
                J2 = st.selectbox("Incorporate J2 Dynamics??",
                                    (" No", "Yes")
                                    )
                input["Include_J2"] = True if J2=="Yes" else False

                input['num_runs'] = st.number_input("Enter Number of models to be trained for each config", 
                                                            min_value=1, max_value=10, value=1)

                with st.form(key='my_form2'):

                    st.write("## Training Set up #")

                    ### literal_eval allows for conversion form string to array ###
                    input['Points'] = literal_eval(st.text_input("Enter Number of traininig points for model training. Enter in form [n, n, n]", 
                                                        value= '[50]', key='Points'))
                                                        
                    input['Polynomial'] = st.multiselect("Orthogonal Polynomials to be used", 
                                                        ("LeP", "CP", "FS"), key='Poly')

                    input['Order'] = literal_eval(st.text_input("Enter Max order of Polynomials to be used for model training. Enter in form [n, n, n]", 
                                                         value= '[10,20,50]', key='Order'))
                        
                    input["run"] = st.form_submit_button("Fit Model")
                

        else:
            ### If Using LambertEq enter appropriate config options ###
            with st.form(key='my_form2'):
                input['num_epochs'] = st.number_input("Enter Number of epochs for model training", 
                                                    min_value=1, max_value=10000, value=1001)

                input['lr'] = st.number_input("Enter learning rate for model training", 
                                                    min_value=1e-5, max_value=2.0, value=2e-2, step=1e-2, format='%.2e')

                input["print_interval"] = st.number_input("Enter Print Interval for display of loss during training", 
                                                    min_value=1, max_value=10000, value=100)
                input["run"] = st.form_submit_button("Fit Model")

        return input