# In this example we first load a pre-trained BERT tokenizer and model.
# We then tokenize our input text. BERT has specific tokenization steps, and the tokenizer takes care of these.
# We perform a forward pass through the model to get the hidden states.
# The size of the output embeddings from the last layer is printed.
# Finally, we use a pipeline for sentiment analysis to classify the sentiment of our input text.

# Please note that the model used in this example ('bert-base-uncased') is a
# pre-trained BERT model and it’s being used directly without any fine-tuning.
# For task-specific applications, you might need to fine-tune the model on your
# specific dataset to achieve better performance. Also, the 'bert-base-uncased'
# model outputs raw scores which are not normalized probabilities. If you need 
# the output to be probabilities, you can apply a softmax function to the scores.


import torch
from transformers import BertTokenizer, BertModel, pipeline

# Load pre-trained model tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Encode text
input_text = "I love machine learning! My favorite library is TensorFlow."
input_tokens = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")

# Load pre-trained model
model = BertModel.from_pretrained('bert-base-uncased')

# Forward pass, get hidden states
with torch.no_grad():
    outputs = model(**input_tokens)

# Only take the output embeddings from the last layer
last_hidden_states = outputs.last_hidden_state
print("Size of the output:", last_hidden_states.size())

# Classification example
classifier = pipeline('sentiment-analysis')
result = classifier(input_text)
print("Sentiment Analysis Result:", result)
