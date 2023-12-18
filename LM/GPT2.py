# In this example we load a pre-trained GPT-2 model and its tokenizer.
# We provide an input text (prompt) to the model.
# The model generates a response based on the input text.
# Finally, we decode the generated tensor back to text and print the result.

# The generate function has several parameters to control the generation process.
# In this example, max_length is set to limit the length of the generated text,
# num_beams is set for beam search, no_repeat_ngram_size is set to avoid repetition,
#  and early_stopping is set to stop the generation when a stopping token is found.

# Please note that GPT-2 is a powerful model, and it requires a good amount of
# computational resources. The example above should work on a standard computer,
# but for larger tasks or longer text generation, you might need more powerful
#  hardware or cloud resources.

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Encode input text (prompt) to tensor
input_text = "Once upon a time, in a land far, far away,"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate text (response) from the model
with torch.no_grad():
    output = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)

# Decode tensor to text
response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

print("Generated Text:")
print(response)
