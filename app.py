#gradio app 

import gradio as gr
import pandas as pd
import pickle
import numpy as np

# 1. Load the Model
with open("Voting_ensemble_model_Best.pkl", "rb") as f:
    model = pickle.load(f)

# 2. The Logic Function
def predict_purchased(Gender, Age, EstimatedSalary):
    
    # Pack inputs into a DataFrame
    # The column names must match your CSV file exactly
    input_df = pd.DataFrame([[
        Gender, Age, EstimatedSalary
    ]],
      columns=[
        'Gender', 'Age', 'EstimatedSalary'
    ])
    
    # Predict
    prediction = model.predict(input_df)[0]
    
    # Return Result
    if prediction == 1:
        return f"Prediction: 1 ----> Means Higher probability to Purchase."
    else:
        return f"Prediction: 0 ---> Means Lower probability to Purchase. "

# 3. The App Interface
# Defining inputs in a list to keep it clean
inputs = [
    gr.Radio(["Male", "Female"], label="Gender"),
    gr.Number(label="Age", value=20),
    gr.Number(label="Estimated Salary (Taka)")
]

app = gr.Interface(
    fn=predict_purchased,
      inputs=inputs,
        outputs="text", 
        title="AdClick Purchase Predictor")

app.launch(share=True)

