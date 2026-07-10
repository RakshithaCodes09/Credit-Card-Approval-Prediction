import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_generator import generate_synthetic_data

def train_model():
    print("Loading/generating credit data...")
    # Load or generate dataset
    data_path = 'data/credit_record.csv'
    if not os.path.exists(data_path):
        os.makedirs('data', exist_ok=True)
        df = generate_synthetic_data(num_records=10000)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)
    
    X = df.drop(columns=['APPROVED', 'FLAG_MOBIL'])
    y = df['APPROVED']
    
    # Identify numeric and categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Categorical features: {categorical_cols}")
    print(f"Numerical features: {numeric_cols}")
    
    # Define preprocessing steps
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    # Combined Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))
    ])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training Gradient Boosting Classifier...")
    pipeline.fit(X_train, y_train)
    
    # Evaluation
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1_score': float(f1_score(y_test, y_pred)),
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'target_approval_rate': float(y.mean())
    }
    
    print("\nModel Evaluation Metrics:")
    for k, v in metrics.items():
        if k != 'confusion_matrix':
            print(f"  {k.upper()}: {v:.4f}")
            
    # Save metrics
    os.makedirs('model', exist_ok=True)
    with open('model/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Save full pipeline
    joblib.dump(pipeline, 'model/model_pipeline.joblib')
    print("Saved pipeline to 'model/model_pipeline.joblib'")
    
    # Extract Feature Importances
    preprocessor_fitted = pipeline.named_steps['preprocessor']
    classifier_fitted = pipeline.named_steps['classifier']
    
    # Get categorical features encoded names
    cat_encoder = preprocessor_fitted.named_transformers_['cat'].named_steps['onehot']
    encoded_cat_cols = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    
    all_features = numeric_cols + encoded_cat_cols
    importances = classifier_fitted.feature_importances_
    
    # Group importances back to original features for cleaner visualization
    feature_imp_df = pd.DataFrame({'feature': all_features, 'importance': importances})
    
    # Map encoded features back to their original names
    grouped_importances = {}
    for _, row in feature_imp_df.iterrows():
        feat = row['feature']
        imp = row['importance']
        original_found = False
        for orig_cat in categorical_cols:
            if feat.startswith(orig_cat + '_'):
                grouped_importances[orig_cat] = grouped_importances.get(orig_cat, 0.0) + imp
                original_found = True
                break
        if not original_found:
            grouped_importances[feat] = grouped_importances.get(feat, 0.0) + imp
            
    # Sort grouped importances
    sorted_importances = sorted(grouped_importances.items(), key=lambda x: x[1], reverse=True)
    sorted_importances_dict = {k: float(v) for k, v in sorted_importances}
    
    with open('model/feature_importances.json', 'w') as f:
        json.dump(sorted_importances_dict, f, indent=4)
    print("Saved feature importances to 'model/feature_importances.json'")
    
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    features_plot = list(sorted_importances_dict.keys())[:10]
    importances_plot = list(sorted_importances_dict.values())[:10]
    sns.barplot(x=importances_plot, y=features_plot, palette='viridis')
    plt.title('Top 10 Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    os.makedirs('static/img', exist_ok=True)
    plt.savefig('static/img/feature_importance.png', dpi=300)
    plt.close()
    
    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm = np.array(metrics['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Rejected', 'Approved'], yticklabels=['Rejected', 'Approved'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('static/img/confusion_matrix.png', dpi=300)
    plt.close()
    
    print("Generated and saved charts in 'static/img/'")

if __name__ == '__main__':
    train_model()
