import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# Model Functions:
###############################################################################################################################################
def xgboost_model(df_countvectorizer_ngrams, df):
    # Define features and target
    X_xgb = df_countvectorizer_ngrams.values
    y_xgb = df['NegoOutcomeLabel'].values
    
    # Split data into training and testing sets
    X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(X_xgb, y_xgb, test_size=0.33, random_state=42, stratify=y_xgb)
    
    # Initial XGBoost model
    xgb_model = xgb.XGBClassifier(random_state=42)
    xgb_model.fit(X_train_xgb, y_train_xgb)
    
    # Predict and evaluate the model
    y_pred_test = xgb_model.predict(X_test_xgb)
    
    # Hyperparameter tuning with GridSearchCV
    parameters = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=parameters, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train_xgb, y_train_xgb)
    
    # Use the best parameters
    best_params = grid_search.best_params_
    best_xgb_model = xgb.XGBClassifier(**best_params, random_state=42, use_label_encoder=False, eval_metric='logloss')
    best_xgb_model.fit(X_train_xgb, y_train_xgb)
    
    # Predict and evaluate the optimized model
    y_pred_best = best_xgb_model.predict(X_test_xgb)
    
    # Results DataFrame
    results_xgb = pd.DataFrame({
        'Model': ['XGBoost', 'XGBoost (Optimized)'],
        'Accuracy': [accuracy_score(y_test_xgb, y_pred_test), accuracy_score(y_test_xgb, y_pred_best)],
        'ROC AUC': [roc_auc_score(y_test_xgb, y_pred_test), roc_auc_score(y_test_xgb, y_pred_best)],
        'F1 Score': [f1_score(y_test_xgb, y_pred_test, average='weighted'), f1_score(y_test_xgb, y_pred_best, average='weighted')]
    })
    
    return results_xgb



def naive_bayes_model(df_countvectorizer_ngrams, df):
    # Define features and target
    X_nb = df_countvectorizer_ngrams.values
    y_nb = df['NegoOutcomeLabel'].values
    
    # Split data into training and testing sets
    X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(X_nb, y_nb, test_size=0.33, random_state=42, stratify=y_nb)
    
    # Initial GaussianNB model
    gnb = GaussianNB()
    gnb.fit(X_train_nb, y_train_nb)
    y_pred_test = gnb.predict(X_test_nb)
    
    # Hyperparameter Tuning
    params = {'var_smoothing': np.logspace(0, -9, num=100)}
    grid_search = GridSearchCV(estimator=GaussianNB(), param_grid=params, scoring='accuracy', cv=10, verbose=1)
    grid_search.fit(X_train_nb, y_train_nb)
    
    # Retrain with best parameters
    gnb_best = GaussianNB(var_smoothing=grid_search.best_params_['var_smoothing'])
    gnb_best.fit(X_train_nb, y_train_nb)
    y_pred_best = gnb_best.predict(X_test_nb)
    
    # Results DataFrame
    results_nb = pd.DataFrame({
        'Model': ['Naive Bayes', 'Naive Bayes (Optimized)'],
        'Accuracy': [accuracy_score(y_test_nb, y_pred_test), accuracy_score(y_test_nb, y_pred_best)],
        'ROC AUC': [roc_auc_score(y_test_nb, y_pred_test), roc_auc_score(y_test_nb, y_pred_best)],
        'F1 Score': [f1_score(y_test_nb, y_pred_test, average='weighted'), f1_score(y_test_nb, y_pred_best, average='weighted')]
    })
    
    return results_nb


def random_forest_model(df_countvectorizer_ngrams, df):
    # Define features and target
    X_rf = df_countvectorizer_ngrams.values
    y_rf = df['NegoOutcomeLabel'].values
    
    # Split data into training and testing sets
    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.33, random_state=42, stratify=y_rf)
    
    # Initialize the model
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_rf, y_train_rf)
    y_pred_rf = rf.predict(X_test_rf)
    
    # Hyperparameter Tuning
    params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=params, scoring='accuracy', cv=10, verbose=1)
    grid_search.fit(X_train_rf, y_train_rf)
    
    # Retrain with best parameters
    rf_best = RandomForestClassifier(**grid_search.best_params_, random_state=42)
    rf_best.fit(X_train_rf, y_train_rf)
    rfc_best_pred = rf_best.predict(X_test_rf)
    
    # Results DataFrame
    results_rf = pd.DataFrame({
        'Model': ['Random Forest', 'Random Forest (Optimized)'],
        'Accuracy': [accuracy_score(y_test_rf, y_pred_rf), accuracy_score(y_test_rf, rfc_best_pred)],
        'ROC AUC': [roc_auc_score(y_test_rf, y_pred_rf), roc_auc_score(y_test_rf, rfc_best_pred)],
        'F1 Score': [f1_score(y_test_rf, y_pred_rf, average='weighted'), f1_score(y_test_rf, rfc_best_pred, average='weighted')]
    })
    
    return results_rf


def logistic_regression_model(df_countvectorizer_ngrams, df):
    # Define features and target
    X_log = df_countvectorizer_ngrams.values
    y_log = df['NegoOutcomeLabel'].values
    
    # Split data into training and testing sets
    X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log, y_log, test_size=0.33, random_state=42, stratify=y_log)
    
    # Initialize and train Logistic Regression model
    logreg = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
    logreg.fit(X_train_log, y_train_log)
    
    # Hyperparameter tuning using GridSearchCV
    parameters = [{'penalty': ['l1', 'l2']}, {'C': [1, 5, 10, 100, 250, 1000]}]
    grid_search = GridSearchCV(estimator=logreg, param_grid=parameters, cv=10, scoring='accuracy')
    grid_search.fit(X_train_log, y_train_log)
    
    # Best Model from GridSearchCV
    best_logreg = grid_search.best_estimator_
    y_pred_best = best_logreg.predict(X_test_log)
    
    # Results DataFrame
    results_logreg = pd.DataFrame({
        'Model': ['Logistic Regression', 'Logistic Regression (Optimized)'],
        'Accuracy': [accuracy_score(y_test_log, y_pred_best), accuracy_score(y_test_log, y_pred_best)],
        'ROC AUC': [roc_auc_score(y_test_log, y_pred_best), roc_auc_score(y_test_log, y_pred_best)],
        'F1 Score': [f1_score(y_test_log, y_pred_best, average='weighted'), f1_score(y_test_log, y_pred_best, average='weighted')]
    })
    
    return results_logreg


def svm_model(df_countvectorizer_ngrams, df):
    # Define features and target
    X_svm = df_countvectorizer_ngrams.values
    y_svm = df['NegoOutcomeLabel'].values
    
    # Split data into training and testing sets
    X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X_svm, y_svm, test_size=0.33, random_state=42, stratify=y_svm)
    
    # Initial SVM model
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(X_train_svm, y_train_svm)
    y_pred_svm = svm.predict(X_test_svm)
    
    # Hyperparameter tuning with GridSearchCV
    parameters = {'C': [1, 10, 100]}
    grid_search = GridSearchCV(estimator=SVC(kernel='linear', random_state=42), param_grid=parameters, scoring='accuracy', cv=10)
    grid_search.fit(X_train_svm, y_train_svm)
    
    # Retrain with the best parameters
    best_svm = grid_search.best_estimator_
    y_pred_best_svm = best_svm.predict(X_test_svm)
    
    # Results DataFrame
    results_svm = pd.DataFrame({
        'Model': ['SVM', 'SVM (Optimized)'],
        'Accuracy': [accuracy_score(y_test_svm, y_pred_svm), accuracy_score(y_test_svm, y_pred_best_svm)],
        'ROC AUC': [roc_auc_score(y_test_svm, y_pred_svm), roc_auc_score(y_test_svm, y_pred_best_svm)],
        'F1 Score': [f1_score(y_test_svm, y_pred_svm, average='weighted'), f1_score(y_test_svm, y_pred_best_svm, average='weighted')]
    })
    
    return results_svm

def compare_models():
    # Call model functions and store the results
    results_xgb = xgboost_model(df_countvectorizer_ngrams, df)
    results_nb = naive_bayes_model(df_countvectorizer_ngrams, df)
    results_rf = random_forest_model(df_countvectorizer_ngrams, df)
    results_logreg = logistic_regression_model(df_countvectorizer_ngrams, df)
    results_svm = svm_model(df_countvectorizer_ngrams, df)
    
    # Combine all model results into a single DataFrame for comparison
    results = pd.concat([results_xgb, results_nb, results_rf, results_logreg, results_svm], ignore_index=True)
    
    return results


# Plot functions
###############################################################################################################################################
def plot_model_comparison(results_combined):
    # Filter for initial models that do not contain parentheses
    initial_models = results_combined[~results_combined['Model'].str.contains(r'\(', case=False)]
    # Filter for optimized models
    optimized_models = results_combined[results_combined['Model'].str.contains('Optimized', case=False)]

    # Plot for Initial Models
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Model', y='Accuracy', data=initial_models, label='Accuracy', marker='o', color='blue', lw=2)
    sns.lineplot(x='Model', y='ROC AUC', data=initial_models, label='ROC AUC', marker='o', color='green', lw=2)
    sns.lineplot(x='Model', y='F1 Score', data=initial_models, label='F1 Score', marker='o', color='red', lw=2)
    plt.title('Initial Model Comparison: Accuracy, ROC AUC, and F1 Score', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    for i in range(len(initial_models)):
        plt.text(i, initial_models['Accuracy'].iloc[i] + 0.005, f'{initial_models["Accuracy"].iloc[i]:.4f}', ha='center', va='bottom')
        plt.text(i, initial_models['ROC AUC'].iloc[i] + 0.005, f'{initial_models["ROC AUC"].iloc[i]:.4f}', ha='center', va='bottom')
        plt.text(i, initial_models['F1 Score'].iloc[i] + 0.005, f'{initial_models["F1 Score"].iloc[i]:.4f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.legend()
    plt.show()

    # Plot for Optimized Models
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Model', y='Accuracy', data=optimized_models, label='Accuracy', marker='o', color='blue', lw=2)
    sns.lineplot(x='Model', y='ROC AUC', data=optimized_models, label='ROC AUC', marker='o', color='green', lw=2)
    sns.lineplot(x='Model', y='F1 Score', data=optimized_models, label='F1 Score', marker='o', color='red', lw=2)
    plt.title('Optimized Model Comparison: Accuracy, ROC AUC, and F1 Score', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    for i in range(len(optimized_models)):
        plt.text(i, optimized_models['Accuracy'].iloc[i] + 0.005, f'{optimized_models["Accuracy"].iloc[i]:.4f}', ha='center', va='bottom')
        plt.text(i, optimized_models['ROC AUC'].iloc[i] + 0.005, f'{optimized_models["ROC AUC"].iloc[i]:.4f}', ha='center', va='bottom')
        plt.text(i, optimized_models['F1 Score'].iloc[i] + 0.005, f'{optimized_models["F1 Score"].iloc[i]:.4f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.legend()
    plt.show()

    # Create a new DataFrame that includes only Accuracy and labels for Base and Optimized models
    base_data = initial_models[['Model', 'Accuracy']]
    base_data['Type'] = 'Base'
    optimized_data = optimized_models[['Model', 'Accuracy']]
    optimized_data['Type'] = 'Optimized'
    combined_data = pd.concat([base_data, optimized_data])

    # Create a pivot table for the stacked bar plot
    pivot_data = combined_data.pivot_table(index='Model', columns='Type', values='Accuracy', aggfunc='max').reset_index()

    # Melt the pivot table to long format for Plotly
    melted_data = pivot_data.melt(id_vars='Model', value_vars=['Base', 'Optimized'], var_name='Type', value_name='Accuracy')

    # Create the interactive stacked bar plot
    fig = px.bar(melted_data, x='Model', y='Accuracy', color='Type', barmode='stack',
                 title='Model Comparison: Accuracy (Base vs Optimized)',
                 labels={'Accuracy': 'Accuracy Score', 'Model': 'Model'},
                 height=600, width=1200)

    # Update layout for better readability
    fig.update_layout(title_font_size=24, xaxis_title_font_size=20, yaxis_title_font_size=20,
                      legend_title_font_size=16, xaxis_tickangle=-45)

    # Show the plot
    fig.show()
