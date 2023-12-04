from transformers import RobertaForSequenceClassification

# Load the model from a directory
model = RobertaForSequenceClassification.from_pretrained("related_url")

import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer

# Initialize the tokenizer and load the fine-tuned model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


# Move the model to the GPU if it's not already there
model.to("cuda")

# Prepare new data
new_text = "Lĩnh vực Công nghệ thông tin: Những người quan tâm đến lĩnh vực Công nghệ thông tin và giải pháp CNTT."
encoded_dict = tokenizer(
    new_text,
    add_special_tokens=True,
    padding="max_length",
    truncation=True,
    max_length=512,
    return_tensors="pt",
)

# Make a prediction
input_ids = encoded_dict["input_ids"].to("cuda")
attention_mask = encoded_dict["attention_mask"].to("cuda")

with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    predicted = torch.argmax(outputs.logits, dim=1)

# Print the predicted label
if predicted.item() == 1:
    print("Related to the topic")
else:
    print("Not related to the topic")
