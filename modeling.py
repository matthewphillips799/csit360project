# modeling.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor

def load_data(target: str = "Obese"):
    # Load the preprocessed data
    wide_df = pd.read_csv("DATA\\obesity.csv")
    
    # Define features and target
    features = ["StateID", "Age", "Education", "Sex", "Income", "Race/Ethnicity"]
    X = wide_df[features]
    y = wide_df[target]
    
    # Remove rows where target is missing
    valid_rows = y.notna()
    X = X[valid_rows]
    y = y[valid_rows]
    
    # Print dataset info
    print(f"\nDataset Info: {len(X)} samples | Target: {target}")
    print("Sample features:")
    print(X.head(2))
    
    return X, y, features

def build_preprocessor(features: list):
    categorical_features = features  # All our features are categorical
    return ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

def evaluate_model(y_true, y_pred, target_name: str):
    metrics = {
        'MAE': {
            'value': mean_absolute_error(y_true, y_pred),
            'interpretation': (
                "Average prediction error in percentage points. "
                "For public health data:\n"
                "  <2% = Excellent\n"
                "  2-3% = Good\n"
                "  >3% = Needs improvement"
            )
        },
        'RMSE': {
            'value': np.sqrt(mean_squared_error(y_true, y_pred)),
            'interpretation': (
                "Similar to MAE but penalizes large errors more.\n"
                "Ideal range:\n"
                "  <3% = Excellent\n"
                "  3-4% = Good\n"
                " >4% = Needs improvement"
            )
        },
        'R²': {
            'value': r2_score(y_true, y_pred),
            'interpretation': (
                "Proportion of variance explained by the model:\n"
                "  0.8-1.0 = Excellent\n"
                "  0.6-0.8 = Good\n"
                "  0.4-0.6 = Moderate\n"
                "  <0.4 = Poor"
            )
        },
        'Explained Variance': {
            'value': explained_variance_score(y_true, y_pred),
            'interpretation': (
                "Similar to R² but less sensitive to outliers.\n"
                "Should be close to R² value."
            )
        }
    }
    
    # Add overall assessment
    mae = metrics['MAE']['value']
    r2 = metrics['R²']['value']
    
    if mae < 2 and r2 > 0.7:
        assessment = "Excellent model for policy decisions"
    elif mae < 3 and r2 > 0.5:
        assessment = "Good model for research purposes"
    else:
        assessment = "Model needs improvement before deployment"
    
    metrics['Overall'] = {'value': None, 'interpretation': assessment}
    
    return metrics

def print_metrics(metrics: dict, model_name: str, target_name: str):
    print(f"\n{model_name.upper()} PERFORMANCE ({target_name})")
    print("===========================================================")
    for metric, data in metrics.items():
        if metric == 'Overall':
            print(f"\n{data['interpretation']}")
            continue
            
        print(f"\n{metric}: {data['value']:.3f}")
        print("-----------------------------------------")
        print(data['interpretation'])
    print("===========================================================")

def train_and_evaluate(X: pd.DataFrame, 
                      y: pd.Series, 
                      preprocessor: ColumnTransformer,
                      model_type: str, 
                      target_name: str):
    # Complete training and evaluation workflow

    # Configure models
    models = {
        'linear': make_pipeline(
            preprocessor,
            StandardScaler(with_mean=False),
            LinearRegression()
        ),
        'ridge': make_pipeline(
            preprocessor,
            StandardScaler(with_mean=False),
            Ridge(alpha=0.5)
        ),
        'random_forest': make_pipeline(
            preprocessor,
            RandomForestRegressor(
                n_estimators=100,
                random_state=123,
                n_jobs=-1)
        ),
        'xgboost': make_pipeline(
            preprocessor,
            XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                random_state=123,
                n_jobs=-1)
        )
    }
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=123
    )
    
    print(f"\nTraining {model_type} model...")
    model = models[model_type]
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, target_name)
    
    # Print results
    print_metrics(metrics, model_type, target_name)
    
    # Feature importance for tree-based models
    if model_type in ['random_forest', 'xgboost']:
        try:
            print("\nTOP PREDICTIVE FACTORS:")
            if model_type == 'random_forest':
                importances = model.named_steps['randomforestregressor'].feature_importances_
            else:
                importances = model.named_steps['xgbregressor'].feature_importances_
            
            # Get feature names
            ohe = model.named_steps['columntransformer'].named_transformers_['cat']
            feature_names = ohe.get_feature_names_out(X.columns)
            
            # Print top 5 features
            for idx in np.argsort(importances)[-5:][::-1]:
                print(f"- {feature_names[idx]}: {importances[idx]:.4f}")
        except Exception as e:
            print(f"\n Could not extract feature importance: {str(e)}")
    
    return model, metrics

def main():
    # Configuration
    TARGETS = {
        'Obese': "Obesity Prevalence (%)",
        'Overweight': "Overweight Prevalence (%)"
    }
    
    MODELS = [
        'linear',
        'ridge', 
        'random_forest',
        'xgboost'
    ]
    
    all_results = {}
    
    print("===========================================================")
    print("PUBLIC HEALTH MODELING: OBESITY/OVERWEIGHT PREDICTORS")
    print("===========================================================")
    
    for target_name, target_desc in TARGETS.items():
        print("===========================================================")
        print(f"ANALYZING: {target_name} ({target_desc})")
        print("===========================================================")
        
        # Load data
        X, y, features = load_data(target_name)
        preprocessor = build_preprocessor(features)
        
        # Train and evaluate models
        target_results = {}
        for model_type in MODELS:
            model, metrics = train_and_evaluate(
                X, y, preprocessor, model_type, target_name
            )
            target_results[model_type] = metrics
        
        all_results[target_name] = target_results
    
    # Final comparison
    print("===========================================================")
    print("FINAL MODEL COMPARISON")
    print("===========================================================")
    
    for target_name in TARGETS:
        print(f"\nBEST MODELS FOR {target_name.upper()}:")
        models = all_results[target_name]
        
        # Find best by MAE
        best_mae = min(models.items(), key=lambda x: x[1]['MAE']['value'])
        print(f"- Best MAE ({best_mae[1]['MAE']['value']:.2f}): {best_mae[0]}")
        
        # Find best by R²
        best_r2 = max(models.items(), key=lambda x: x[1]['R²']['value'])
        print(f"- Best R² ({best_r2[1]['R²']['value']:.2f}): {best_r2[0]}")

if __name__ == "__main__":
    main()