import re

# Dữ liệu đầu vào (ví dụ)
f = open("bing.json", "r", encoding="utf-8")
data = f.read()
print(data)

# Sử dụng biểu thức chính quy để trích xuất các URL
url_pattern = r"https?://\S+"
urls = re.findall(url_pattern, data)

# In danh sách các URL đã trích xuất
for url in urls:
    print(url)
