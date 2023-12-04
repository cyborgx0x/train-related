import torch
from transformers import RobertaModel, RobertaTokenizer

# Khởi tạo tokenizer và mô hình RoBERTa
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base")

# Dữ liệu đầu vào
keyword = "2 Casino Ransomware Attacks: Caesars, MGM"
text = "MGM reeling from cyber 'chaos' 5 days after attack as Caesars ..."

# Tokenize và biểu diễn từ khóa và đoạn văn bản
input_ids_keyword = tokenizer.encode(
    keyword, add_special_tokens=False, return_tensors="pt"
)
input_ids_text = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")

# Biểu diễn nhúng cho từ khóa và đoạn văn bản
embedding_keyword = model(input_ids_keyword).last_hidden_state.mean(dim=1)
embedding_text = model(input_ids_text).last_hidden_state.mean(dim=1)

# Tính toán sự tương đồng cosine
similarity = torch.nn.functional.cosine_similarity(embedding_keyword, embedding_text)

print(f"Similarity score: {similarity.item()}")
