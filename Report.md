#   Project Report

##  Approach and Key Decisions

This project explored two primary approaches for modeling phoneme duration: an LSTM-based model and a decision tree-based model. The LSTM model utilized PyTorch to construct a multi-layer LSTM network that captures the temporal dependencies in phoneme sequences. The embedding dimension was set to 32, and the hidden dimension to 64, with a dropout rate of 0.2 to mitigate overfitting. For the decision tree model, a RandomForestRegressor from scikit-learn was chosen due to its ability to handle non-linear relationships and provide valuable feature importance metrics.

##   Challenges and Solutions

A significant challenge encountered was the efficient handling of a large dataset. This was addressed by utilizing data loaders in PyTorch for the LSTM model and preprocessed data files for the decision tree model. Another issue was a version inconsistency warning from scikit-learn, which was noted but not resolved within the project timeframe.

##  Insights from Data and Model Performance

The LSTM model demonstrated strong performance, achieving a mean squared error (MSE) of 0.0032 on the test set. The decision tree model provided valuable insights into the differences in phoneme duration between native and non-native speakers, particularly for phonemes like 'l;A' and 'y;A'.

##  Time Breakdown and Tools Used

The project was completed in a total of 8 hours, with the following time allocation: 2 hours on feature processing and engineering, 2 hours on the decision tree model, 2 hours on the LSTM model, and 2 hours on code optimization and documentation.

#   Future Improvements

Future work could explore the use of Transformer-based architectures for improved model performance. Additionally, pre-training on larger datasets may lead to more accurate phoneme feature embeddings.
