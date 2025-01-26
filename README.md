# Transformer-Based Models Implementation

## Overview
This chapter focuses on the implementation of Transformer-based models, with a particular emphasis on various BERT models. The motivation for this choice stems from several key factors:
- **Task Alignment**: BERT's tasks, Next Sentence Prediction and Masked Language Modeling (Amatriain et al., 2024), align closely with our objective: predicting negotiation outcomes.
- **Open-Source Availability**: BERT is an open-source model and one of the earliest Transformer-based architectures, making it widely accessible.
- **Abundance of Pretrained Models**: Numerous fine-tuned BERT models are readily available, such as those on [Huggingface](https://huggingface.co/).
- **Dominance in Text Classification**: The sheer number of BERT models outpaces other Transformer-based models in the text classification category.

---

## Challenges and Focus
While efforts were made to implement other Transformer models, such as GPT, hardware limitations posed significant challenges:
- Even with access to the **BW-Uni-Cluster**, memory errors occurred due to high computational demands.
- Limited cluster access time further constrained experimentation.

As a result, the focus shifted to BERT models, which were more feasible given the hardware and time constraints.

---

## Model Implementation
### Key Points
1. **File Structure**: The files for the different models are organized similarly, varying primarily in the specific model used.
2. **Training Runs**: Each model was trained twice:
   - **5 epochs**
   - **10 epochs**
3. **Storage**: Each training run is stored in a separate file due to memory and timeslot limitations on the cluster.
4. **File Naming**: Files are named after the model and the number of epochs (e.g., `DistilBert_5epochs.ipynb` and `DistilBert_10epochs.ipynb`).
5. **Explanations**: Each file contains:
   - A brief explanation of the model and rationale for its use at the top.
   - Results and interpretation of the model's performance at the bottom.

### Hyperparameter Tuning
Due to computational constraints, hyperparameter tuning was not feasible.

---

## Models Used
The following five BERT-based models were implemented in the specified order:
1. **DistilBert** (5 epochs, 10 epochs)
2. **DistilRoberta-base** (5 epochs, 10 epochs)
3. **Finbert** (5 epochs, 10 epochs)
4. **DistilRoberta-financial** (5 epochs, 10 epochs)
5. **Finbert-Sentiment** (5 epochs, 10 epochs)

- Each file includes observations unique to the model and its results.

---

## Results and Evaluation
- The results for all models were saved and analyzed in the file `Transformers_comparison.ipynb`.
- **Key Considerations**:
  - The computational effort required for training these models was significant.
  - Training was only possible due to access to the **BW-Uni-Cluster**.

---

## Folder and File Descriptions

### **Folders**
- **`evaluation_files`**: Contains the saved performances of the different models.
- **`df_complete_clensing`**: This dataset is identical to `clensed_data` but renamed for clarity.

### **Files**
- Individual files for each model and training run (e.g., `DistilBert_5epochs.ipynb`).
- `Transformers_comparison.ipynb`: A comprehensive comparison of model performances.

---

## Notes
- Ensure all files are well-documented for clarity and reproducibility.
- High-performance hardware is recommended for running these models due to their computational demands.

---
