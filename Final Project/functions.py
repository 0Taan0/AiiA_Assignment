###############################################################################################################################################
# Compare Plot functions
###############################################################################################################################################
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import seaborn as sns

def plot_model_comparison(results_combined):
    """
    Plot the comparison of initial and optimized models based on Accuracy, ROC AUC, and F1 Score.
    
    Parameters:
    results_combined (pd.DataFrame): DataFrame containing the results of the models.
    """
    # Filter for initial models that do not contain parentheses
    initial_models = results_combined[~results_combined['Model'].str.contains(r'\(', case=False)]
    # Filter for optimized models
    optimized_models = results_combined[results_combined['Model'].str.contains('Optimized', case=False)]

    # Plot for Initial Models
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Model', y='Accuracy', data=initial_models, label='Accuracy', marker='o', color='teal', lw=2)
    sns.lineplot(x='Model', y='ROC AUC', data=initial_models, label='ROC AUC', marker='o', color='purple', lw=2)
    sns.lineplot(x='Model', y='F1 Score', data=initial_models, label='F1 Score', marker='o', color='cyan', lw=2)
    plt.title('Initial Model Comparison: Accuracy, ROC AUC, and F1 Score', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    for i in range(len(initial_models)):
        plt.text(i, initial_models['Accuracy'].iloc[i] + 0.005, f'{initial_models["Accuracy"].iloc[i]:.4f}', ha='center', va='bottom')
        plt.text(i, initial_models['ROC AUC'].iloc[i] + 0.005, f'{initial_models["ROC AUC"].iloc[i]:.4f}', ha='center', va='bottom')
        plt.text(i, initial_models['F1 Score'].iloc[i] + 0.005, f'{initial_models["F1 Score"].iloc[i]:.4f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    # Plot for Optimized Models
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Model', y='Accuracy', data=optimized_models, label='Accuracy', marker='o', color='teal', lw=2)
    sns.lineplot(x='Model', y='ROC AUC', data=optimized_models, label='ROC AUC', marker='o', color='purple', lw=2)
    sns.lineplot(x='Model', y='F1 Score', data=optimized_models, label='F1 Score', marker='o', color='cyan', lw=2)
    plt.title('Optimized Model Comparison: Accuracy, ROC AUC, and F1 Score', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    for i in range(len(optimized_models)):
        plt.text(i, optimized_models['Accuracy'].iloc[i] + 0.005, f'{optimized_models["Accuracy"].iloc[i]:.4f}', ha='center', va='bottom')
        plt.text(i, optimized_models['ROC AUC'].iloc[i] + 0.005, f'{optimized_models["ROC AUC"].iloc[i]:.4f}', ha='center', va='bottom')
        plt.text(i, optimized_models['F1 Score'].iloc[i] + 0.005, f'{optimized_models["F1 Score"].iloc[i]:.4f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
###############################################################################################################################################
# Preprocessing dataframes function
###############################################################################################################################################
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, TreebankWordTokenizer

# Ensure necessary NLTK resources are downloaded
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize necessary components
lemmatizer = WordNetLemmatizer()
stopwords_en = set(stopwords.words('english'))
regexp_tokenizer = RegexpTokenizer(r'\w+')
treebank_tokenizer = TreebankWordTokenizer()

def preprocess_text(df, text_column):
    """
    Preprocess the text data in the specified column of the dataframe.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the text data.
    text_column (str): The name of the column containing the text data.
    
    Returns:
    pd.DataFrame: The dataframe with the preprocessed text data.
    """
    
    def lemmatize_text(text):
        return " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    
    def remove_stopwords(words):
        return [w for w in words if w.lower() not in stopwords_en]
    
    def combine_text(list_of_text):
        return ' '.join(list_of_text)
    
    # Step 1: Lemmatization
    df[text_column] = df[text_column].apply(lemmatize_text)
    
    # Step 2: Tokenization
    df[text_column] = df[text_column].apply(lambda x: regexp_tokenizer.tokenize(x))
    
    # Step 3: Remove Stopwords
    df[text_column] = df[text_column].apply(remove_stopwords)
    
    # Step 4: Combine Text
    df[text_column] = df[text_column].apply(combine_text)
    
    return df


###############################################################################################################################################
##Install Packages Function 
###############################################################################################################################################
import subprocess
import sys

def install_and_import_packages():
    """
    Install and import necessary packages for the project.
    """
    packages = {
        "numpy": "np",
        "pandas": "pd",
        "matplotlib.pyplot": "plt",
        "seaborn": "sns",
        "nltk": "nltk",
        "openpyxl": None,
        "string": "string",
        "re": "re",
        "warnings": "warnings"
    }
    
    for package, alias in packages.items():
        try:
            if alias:
                globals()[alias] = __import__(package)
            else:
                __import__(package)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])
            if alias:
                globals()[alias] = __import__(package)
            else:
                __import__(package)
    
    # Suppress warnings
    warnings.filterwarnings('ignore')
###############################################################################################################################################

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_overall_findings(results):
    """
    Plot the overall findings of each representative technique (Bag of Words, TF-IDF, Word Embeddings)
    for the standard models and also with RNN and transformers.
    
    Parameters:
    results (dict): A dictionary where keys are the names of the techniques and values are dataframes
                    containing the performance metrics for each model.
    """
    # Create a figure with subplots
    fig, axes = plt.subplots(1, len(results), figsize=(20, 5), sharey=True)
    
    # Plot each technique's results
    for ax, (technique, df) in zip(axes, results.items()):
        sns.barplot(x='Model', y='Score', hue='Metric', data=df, ax=ax)
        ax.set_title(technique)
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.legend(loc='upper right')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()

# Example usage
# Assuming you have dataframes for each technique with columns ['Model', 'Metric', 'Score']
# results = {
#     'Bag of Words': df_bow,
#     'TF-IDF': df_tfidf,
#     'Word Embeddings': df_word_embeddings,
#     'RNN': df_rnn,
#     'Transformers': df_transformers
# }
# plot_overall_findings(results)
