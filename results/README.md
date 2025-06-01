## Results

Here we store the results of the experiments conducted to train and evaluate the performance of our deep learning model.

The results are organized as follows:

- `training_simple/` contains the results for the simple model trained using the CLIP and finetuned CLIP embeddings and then tested on the regular test set. (Using the notation of the report this is test set (1)).

- `train_normal_test_oh_shuffled/` contains the results of the simple models trained using CLIP and finetuned CLIP embeddings, then tested on the test dataset with one of the modalities of hateful content swapped.

- `robust_training/` contains the results of the robust training process (used to train the models including the examples with swapped modalities in the training set). These are the models have already been trained using CLIP or finetuned CLIP embeddings, which we train further for a few epochs including the examples where for the hateful videos we swap one of the modalities (frames or text) with non-hateful ones. We then either evaluate their performance on the regular test set ((1) from the report) or on the test sets with text or images swapped in the same way as the training set (sets (2) and (3) from report).