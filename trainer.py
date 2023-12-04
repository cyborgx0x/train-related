import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import AdamW, RobertaForSequenceClassification, RobertaTokenizer

# Initialize the RoBERTa tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base")

# Define your data (text) and labels (related or not related)
texts = [
    "Chiến tranh Dải Gaza 2023 đã gây ra nhiều thiệt hại nghiêm trọng.",
    "Tensions rise in Gaza as the conflict escalates.",
    "Peace negotiations are underway in an attempt to resolve the Gaza crisis.",
    "The international community is closely monitoring the situation in Gaza.",
    "Residents in Gaza are seeking refuge in shelters to escape the conflict.",
    "The UN calls for an immediate ceasefire in Gaza.",
    "The humanitarian crisis in Gaza worsens with each passing day.",
    "Gaza conflict: A look at the key events and developments.",
    "Israeli airstrikes target key locations in Gaza.",
    "Palestinian leaders call for international intervention in the Gaza crisis.",
    "The impact of the Gaza conflict on civilian populations is devastating.",
    "Gaza conflict: The role of neighboring countries in mediating peace.",
    "The world reacts to the ongoing violence in Gaza.",
    "Gaza hospitals overwhelmed as casualties continue to rise.",
    "Gaza conflict: A timeline of the events leading up to the crisis.",
    "Calls for a ceasefire in Gaza grow louder from international leaders.",
    "Israel and Hamas exchange fire in the ongoing conflict.",
    "Gaza crisis: Efforts to provide humanitarian aid to those in need.",
    "Gaza conflict: The effects of the crisis on children and families.",
    "The economic impact of the Gaza conflict on the region.",
]

labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

unrelated_texts = [
    "Hôm nay là một ngày nắng đẹp.",
    "Sân khấu biểu diễn âm nhạc tại thành phố đã sẵn sàng cho buổi biểu diễn tối nay.",
    "Những quy tắc an toàn giao thông cần được tuân thủ mỗi khi tham gia giao thông đường bộ.",
    "Bão dự kiến sẽ đổ bộ vào khu vực biển phía Đông trong thời gian tới.",
    "Thị trường chứng khoán đang trải qua biến động lớn.",
    "Mùa hè là thời gian lý tưởng để thư giãn và du lịch.",
    "Các nhà khoa học đã phát hiện một loài cây mới ở rừng rậm Amazon.",
    "Sự kiện thể thao quốc tế đã thu hút hàng ngàn người tham gia.",
    "Những trò chơi video mới ra mắt đang tạo cơn sốt trên thị trường.",
    "Nhà hàng mới khai trương tại khu vực trung tâm thành phố.",
    "Bóng đá là môn thể thao phổ biến trên toàn thế giới.",
    "Thế giới động vật hoang dã là một chủ đề thú vị để nghiên cứu.",
    "Công nghệ mới giúp tăng cường hiệu suất làm việc.",
    "Cuộc thi nấu ăn lớn đã tổ chức tại thành phố cuối tuần qua.",
    "Chương trình truyền hình mới đã ra mắt trên các kênh truyền hình.",
    "Sách mới của tác giả nổi tiếng đang được đánh giá cao.",
    "Học hỏi là một phần quan trọng của cuộc sống của chúng ta.",
    "Sức khỏe là tài sản quý báu mà chúng ta cần bảo vệ.",
    "Thời tiết trong tuần tới dự kiến sẽ ổn định.",
    "Điều hòa không khí là một thiết bị quan trọng trong mùa hè nắng nóng.",
]

unrelated_labels = [0] * 20
texts = texts + unrelated_texts
labels = labels + unrelated_labels


# Tokenize and prepare the data for the model
input_ids = []
attention_masks = []
for text in texts:
    encoded_dict = tokenizer(
        text,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    input_ids.append(encoded_dict["input_ids"])
    attention_masks.append(encoded_dict["attention_mask"])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)
# Move the model to the GPU
model.to("cuda")

# Create a TensorDataset and DataLoader
dataset = TensorDataset(
    input_ids.to("cuda"), attention_masks.to("cuda"), labels.to("cuda")
)
print(dataset)
# Split the data into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
batch_size = 2
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define the optimizer and loss function
optimizer = AdamW(model.parameters(), lr=1e-5)

# Fine-tune the model
epochs = 3
for epoch in range(epochs):
    model.train()
    for batch in train_dataloader:
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in val_dataloader:
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask)
        predicted = torch.argmax(outputs.logits, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

model.save_pretrained("related_url")
# Prepare new data
new_text = "Chiến tranh Dải Gaza 2023"
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
