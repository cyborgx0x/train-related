import torch
from transformers import RobertaModel, RobertaTokenizer

# Load the pre-trained RoBERTa model and tokenizer
model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained("related_url")

# Get the vocabulary as a list of tokens
vocabulary = list(tokenizer.get_vocab().keys())

# Iterate through the vocabulary and get the weights for each token
token_weights = {}
for token in vocabulary:
    token_id = tokenizer.encode(token, add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        output = model(token_id)
    token_weights[token] = output.last_hidden_state.mean().item()

# Print tokens and their weights
for token, weight in token_weights.items():
    print(f"Token: {token}, Weight: {weight}")
