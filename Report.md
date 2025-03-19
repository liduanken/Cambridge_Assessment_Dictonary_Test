# Project Report

## Approach and Key Decisions

In this project, I explored two main approaches for modeling phoneme duration: an LSTM-based model and a decision tree-based model. For the LSTM model, I used PyTorch to build a multi-layer LSTM network that captures the temporal dependencies in phoneme sequences. I decided to set the embedding dimension to 32 and the hidden dimension to 64, with a dropout rate of 0.2 to help prevent overfitting. For the decision tree model, I went with a RandomForestRegressor from scikit-learn because it's great at handling non-linear relationships and provides useful feature importance metrics.

## Challenges and Solutions

One of the big challenges I faced was efficiently handling the large dataset. I tackled this by using data loaders in PyTorch for the LSTM model and preprocessed data files for the decision tree model. Another issue was a version inconsistency warning from scikit-learn, which I noted but didn't have time to resolve.

## Insights from Data and Model Performance

The LSTM model performed well, achieving a mean squared error (MSE) of 0.0032 on the test set, which I was quite happy with. The decision tree model gave me some interesting insights into the differences in phoneme duration between native and non-native speakers, especially for phonemes like 'l;A' and 'y;A'.

## Time Breakdown and Tools Used

I completed the project in a total of 8 hours. I spent 2 hours on feature processing and engineering, another 2 hours on the decision tree model, 2 hours on the LSTM model, and the last 2 hours on optimizing the code structure and writing documentation.

## Future Improvements

Looking ahead, I think it would be exciting to use Transformer-based architectures to build even better models. Plus, pre-training on more data could help me derive more accurate phoneme feature embeddings.
