import pandas as pd
import numpy as np

def clean_data():

    # Load data
    df = pd.read_csv(
        "DATA\\Nutrition__Physical_Activity__and_Obesity_-_Behavioral_Risk_Factor_Surveillance_System.csv"
    )
    
    # Columns to keep
    keep_cols = [
        "YearStart", "LocationAbbr", "Age(years)", "Education", 
        "Sex", "Income", "Race/Ethnicity", "QuestionID", "Data_Value"
    ]
    df = df[keep_cols]
    
    # Rename columns
    df = df.rename(columns={
        "YearStart": "Year",
        "LocationAbbr": "StateID",
        "Age(years)": "Age",
        "Data_Value": "Percentage"
    })
    
    # Filter to only keep obesity and overweight questions
    df = df[df["QuestionID"].isin(["Q036", "Q037"])]
    
    # Map question IDs to readable names
    question_rename = {
        "Q036": "Obese",
        "Q037": "Overweight"
    }
    df["Question"] = df["QuestionID"].map(question_rename)
    
    # Fill missing demographic data with "Unknown"
    demo_cols = ["Age", "Education", "Sex", "Income", "Race/Ethnicity"]
    df[demo_cols] = df[demo_cols].fillna("Unknown")
    
    
    # Create wide format
    df = df.pivot_table(
        index=["Year", "StateID"] + demo_cols,
        columns="Question",
        values="Percentage",
        aggfunc="mean"
    ).reset_index()
    
    df.to_csv("DATA\\obesity.csv", index=False)
    
    return  df

if __name__ == "__main__":
    print("Processing data...")
    df = clean_data()
    print("\nSample:")
    print(df.head())