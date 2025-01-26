This chapter focuses on the implementation of Transformer-based models, with a particular emphasis on different BERT models. This choice is motivated by several reasons. 
One aspect is that BERT's tasks are classified as Next Sentence Prediction and Masked Language Modeling (Amatriain et al., 2024), which align with our objective: predicting negotiation outcomes. 
Additionally, BERT is an open-source model and one of the earliest Transformer-based models. As a result, there are numerous fine-tuned models available, for example, on Huggingface. 
Furthermore, the sheer number of BERT models outweighs other models in the text classification category.

It is important to note that we attempted to implement other Transformer types, such as GPT models. However, during this process, hardware limitations became evident. Even when using the BW-Uni-Cluster, memory errors occurred. 
Due to these constraints and the limited access time on the BW-Uni-Cluster, which is required for computation, the focus was placed on BERT models.

The files for the different models are structured similarly, with only the models themselves varying. 
Additionally, due to the computational effort involved, hyperparameter tuning was not feasible. 
A total of five different BERT-based models were used, with each model being trained twice: once for 5 epochs and once for 10 epochs. 
Each training run is stored in a separate file, as the cluster's memory was limited, as were the available timeslots. 
Each file is named after the model and the number of epochs. At the top of each file, a brief explanation of the model and the rationale for using it is provided (this explanation is identical for both the 5-epoch and 10-epoch files per model). 
At the bottom, the results for each model are interpreted.

The results of all models were saved, and a comparison was conducted in the file Transformers_comparison.ipynb. It should be noted that the computational effort required for these models is very high. 
Training these models was only possible thanks to the BW-Uni-Cluster.

The order of the model was 1. DistilBert (5-epochs, 10 epochs), 2. DistilRoberta-base (5-epochs, 10 epochs), 3. Finbert (5-epochs, 10 epochs), 4. DistilRoberta-financial (5-epochs, 10 epochs), 5. Finbert-Sentiment (5-epochs, 10 epochs). 
With each file only new oberservations will be commented on. 

df_complete_clensing is identical to clensed_data but renamed.

The folder evaluation_files contains the saved performences of the differnt models. 