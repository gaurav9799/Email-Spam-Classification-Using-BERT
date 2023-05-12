# Email-Spam-Classification-Using-BERT

This repository contains code for classifying emails as spam or not using BERT (Bidirectional Encoder Representations from Transformers). BERT is a powerful language model developed by Google that leverages contextualized word embeddings to capture the meaning of words based on their surrounding context.

## Dataset

The dataset used for this classification task is available in the spam.csv file. It consists of labeled email messages, where each message is tagged as either spam or ham (not spam). The dataset is preprocessed and split into training and testing sets.

## Requirements

The code is implemented in Python and utilizes the following libraries:

- TensorFlow
- TensorFlow Hub
- TensorFlow Text
- Pandas
- NumPy
- Scikit-learn

## Model Training and Evaluation

The notebook contains the code for training and evaluating the BERT-based email spam classification model. It performs the following steps:

- Data loading and preprocessing
- Splitting the dataset into training and testing sets
- Building the BERT model architecture
- Compiling and training the model
- Evaluating the model's performance on the testing set
- Applying the model to classify custom email messages

## Results

After training the model, you will obtain evaluation metrics such as accuracy, precision, recall, and a confusion matrix. These metrics provide insights into the model's performance in classifying spam and non-spam emails.

## Contributing

Contributions to this repository are always welcome. If you find any issues or have suggestions for improvements, please feel free to submit a pull request.

## Acknowledgments

- The code implementation for email spam classification is based on the concepts and techniques from the field of natural language processing.
- BERT (Bidirectional Encoder Representations from Transformers) is a language model developed by Google. Its implementation is provided by TensorFlow Hub.
