import streamlit as st
import re

# Define possible optimizers in a dict.
# Format: optimizer -> default learning rate
OPTIMIZERS = {
    "Adam": 0.001,
    "Adadelta": 1.0,
    "Adagrad": 0.01,
    "Adamax": 0.002,
    "RMSprop": 0.01,
    "SGD": 0.1,
}

def show():
    """Shows the sidebar components for the template and returns user input as dict."""

    input = {}

    with st.sidebar:

    #------------------------------------------------------------------------

        st.write("# Model Architectures")
        input["number_of_models"] = st.number_input("How many Models do you want to train?", min_value=1, max_value=10, step=1)

        collect_numbers = lambda x : [int(i) for i in re.split("[^0-9]", x) if i != ""]
        input["optimizer"] = {}
        input["batch_size"] = {}
        input["lr"] = {}
        input["Layer_Units"] = {}
        input["epochs"] = {}
    
        for i in range(int(input["number_of_models"])):
            st.write(f"## Model Architecture {i+1}")
            input["optimizer"][i] = st.selectbox("Optimizer ", list(OPTIMIZERS.keys()) ,key = 'Model-opt'+str(i))
            input["batch_size"][i] = st.number_input("Batch Size", min_value=1, value=100,  format="%i", key = 'Model-bs'+str(i))
            input["lr"][i] = st.number_input("Learning Rate", min_value=0.0, max_value=1.0, value=0.01, format="%f", key = 'Model-lr'+str(i))

            layterunits  = st.text_input("Layer Units, last layer (output) excluded", key = 'Model-lu'+str(i))
            input["Layer_Units"][i] =  collect_numbers(layterunits)
            input["epochs"][i] = st.number_input("Epochs", min_value=0, max_value=None, value=10, format="%i", key = 'Model-ep'+str(i))
        
        st.write(f"## General Model Hyperparameters")
        input["Batch_Normalisation"] = st.selectbox("Batch Normalisation", ["on", "off"])
        input["dropout"] = st.number_input("Dropout Rate - if none enter 0")
        input["activation_func"] = st.selectbox("Select Activation Function", [None,"relu", "sigmoid", "tanh"])
        input["loss_metric"] = st.selectbox("Loss Metric", ["KLD", "MAE", "MAPE", "MSE", "binary_crossentropy", "categorical_crossentropy"])
        input["tracking_metric"] = st.selectbox("Tracking Metric", ["KLD", "MAE", "MAPE", "MSE", "binary_crossentropy", "categorical_crossentropy"])

        
    #------------------------------------------------------------------------

        st.write("## Input data")
        input["data_format"] = st.selectbox(
            "Which data do you want to use?",
            ("Public dataset", "Numpy arrays"),
        )
        if input["data_format"] == "Numpy arrays":
            st.write(
                """
            Expected format: `[features, label]`
            - `features` has array shape (num samples, num features)
            - `labels` has array shape (num samples, )
            """
            )

    #------------------------------------------------------------------------
        st.write("## Model Saving and Callbacks")
        input["save_losses"] = st.selectbox("Save Model Parameters and logs", ["on", "off"])

        input["save_model"] = st.selectbox("Save Model if performance threshold achieved", ["on", "off"])
        if input["save_model"]=="on":
            input["threshold_type"] = st.selectbox("Performance Threshold type", ["min", "max"])
            input["save_threshold"] = st.number_input("Performance Threshold eg. 0.25")
        
        input["include_callbacks"] = st.selectbox("Callbacks", ["on", "off"])
        if input["include_callbacks"] == "on":

            st.write("Learning Rate Reduction")
            input["reduce_lr_on_plateau"] = st.selectbox("Learning Rate Reduction if performance reaches plateau", ["on", "off"])
            if input["reduce_lr_on_plateau"] == "on":

                input["lr_reduction_factor"] = st.number_input('lr reduction factor')
                input["lr_min_loss_delta"] = st.number_input('lr min loss delta')
                input["lr_patience"] = st.number_input('lr patience')
                input["min_lr"] = st.number_input('minimum lr')

            st.write("Early Stopping")
            input["early_stopping"] = st.selectbox("Early Stopping if performance reaches pleateau", ["on", "off"])
            if input["early_stopping"] == "on":
                input["min_loss_delta"] = st.number_input('min loss delta')
                input["early_stopping_patience"] = st.number_input("Early Stopping patience")

        st.write("## Visualisation Tools")
        input["visualise_data"] = st.selectbox("Visualise Data", ["on", "off"])
        if input["visualise_data"] == "on":
            input["feature_to_visualise"] = st.number_input("Select feature column to visualise")
        input["visualization_tool"] = st.selectbox("Visualisation Tool", ["Tensorbaord", "Weights & Biases", "comet.ml" , "Aim"])
        
            

    return input
            

if __name__ == "__main__":
    show()