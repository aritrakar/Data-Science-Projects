# Job Rating Prediction using NLP

## Problem Statement

The problem is to predict the job rating of a company based on the reviews given by the employees. The dataset is taken from Glassdoor. The dataset contains the following columns:

- overall_rating: Scale of 1-5
- firm: Name of the company
- date_review: Date of the review
- job_title: Job title of the employee
- location: Location of the company
- headline: Headline of the review
- pros: Pros of the company
- cons: Cons of the company
- year: Year of the review
- small: Whether the data point belongs to the small dataset or not

## Methodology

In the first notebook, the features used are straightforward:

1. Sentiment score for pros
2. Sentiment score for cons
3. Encoded firm
4. Length for pros
5. Length for cons

Then, a simple multi-input neural network is trained on the above inputs:

```
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_1 (InputLayer)           [(None, 1)]          0           []                               
                                                                                                  
 input_2 (InputLayer)           [(None, 1)]          0           []                               
                                                                                                  
 input_3 (InputLayer)           [(None, 1)]          0           []                               
                                                                                                  
 input_4 (InputLayer)           [(None, 1)]          0           []                               
                                                                                                  
 input_5 (InputLayer)           [(None, 1)]          0           []                               
                                                                                                  
 dense (Dense)                  (None, 16)           32          ['input_1[0][0]']                
                                                                                                  
 dense_1 (Dense)                (None, 16)           32          ['input_2[0][0]']                
                                                                                                  
 dense_2 (Dense)                (None, 16)           32          ['input_3[0][0]']                
                                                                                                  
 dense_3 (Dense)                (None, 16)           32          ['input_4[0][0]']                
                                                                                                  
 dense_4 (Dense)                (None, 16)           32          ['input_5[0][0]']                
                                                                                                  
 concatenate (Concatenate)      (None, 80)           0           ['dense[0][0]',                  
                                                                  'dense_1[0][0]',                
                                                                  'dense_2[0][0]',                
                                                                  'dense_3[0][0]',                
                                                                  'dense_4[0][0]']                
                                                                                                  
 dense_5 (Dense)                (None, 16)           1296        ['concatenate[0][0]']            
                                                                                                  
 dense_6 (Dense)                (None, 1)            17          ['dense_5[0][0]']                
                                                                                                  
==================================================================================================
Total params: 1,473
Trainable params: 1,473
Non-trainable params: 0
```

In the second notebook, a more complex approach is used. First, the data is preprocessed, same as before. Next, word embeddings are generated using the GloVe word embeddings. Then, PCA is performed to reduce the dimensionality. Finally, a multi-input neural network is trained on the following inputs:

```bash
MyNN(
  (fc1): Linear(in_features=150, out_features=200, bias=True)
  (relu): ReLU()
  (dropout): Dropout(p=0.5, inplace=False)
  (fc2): Linear(in_features=200, out_features=200, bias=True)
  (fc3): Linear(in_features=200, out_features=200, bias=True)
  (fc4): Linear(in_features=200, out_features=200, bias=True)
  (output): Linear(in_features=200, out_features=5, bias=True)
)
```

I experimented with using LSTMs, but unfortunately, they took a lot of time to train and the results were not satisfactory. The final model is a much simpler multi-input neural network.
