import re
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pylatexenc.latex2text import LatexNodes2Text
from torchviz import make_dot
import matplotlib.pyplot as plt
import networkx as nx
from pylatexenc.latexwalker import LatexWalker
from TexSoup import TexSoup
from py_linq import Enumerable


# Функция для извлечения секций из TeX-документа с использование регулярных выражений
def extract_sections_regex(tex_content):
    sections = re.findall(r'\\section{(.*?)}', tex_content)  # Извлекаем заголовки
    contents = re.split(r'\\section{.*?}', tex_content)[1:]  # Разбиваем по секциям
    return list(zip(sections, contents))

# Функция для извлечения секций из TeX-документа с использование библиотеки pylatexenc
def extract_sections_lib(tex_content):
    sections = {}
    current_section = "Unknown"
    buffer = []
    for line in tex_content.split("\n"):
        if line.startswith("\\section{"):
            if buffer:
                sections[current_section] = " ".join(buffer)
                buffer = []
            current_section = LatexNodes2Text().latex_to_text(line.strip().split("{")[1].split("}")[0])
        else:
            buffer.append(LatexNodes2Text().latex_to_text(line.strip()))
    if buffer:
        sections[current_section] = " ".join(buffer)
    return list(sections.items())


#### tests

file_name = "G:/UserData/Desktop/disser/TeX-collection-sections-fix/TeX-collection/Abyzov.tex"
file_content = ""
with open(file_name, "r", encoding="utf-8") as f:
    file_content = f.read()



#latexwalker = LatexWalker(file_content)
#(nodelist, pos, len_) = latexwalker.get_latex_nodes(pos=0)

#abstract = list(filter(lambda node: node.macroname == "abstract", nodelist))
#abstract = [obj for obj in nodelist if getattr(obj, 'macroname', None) == 'abstract']
#introduction = list(filter(lambda node: node.macroname == "section", nodelist))

#(nodelist[0].macroname)

standalone_block_names = ['title', 'abstract', 'author', 'email', 'shorttit', 'keywords', ]
bibliography_block_name = 'thebibliography'
section_block_names = ['Introduction', 'conclusion']

def extract_standalone(texsoup_item):
    result = {}
    error_blocks = []
    for i, block in enumerate(standalone_block_names):
        found = texsoup_item.find(block)
        if found is None:
            error_blocks.append(block)
        else:
            block_content_buffer = ''.join(map(lambda x: str(x).replace('\n', ' '), found.contents))
            block_content_buffer = block_content_buffer.strip()
            result[block] = block_content_buffer
            #merge contents if obj is texnode
    return result, error_blocks

def extract_bibliography(texsoup_item):
    bibliography_block_name = 'thebibliography'
    found = texsoup_item.find(bibliography_block_name)
    return found


texsoup = TexSoup(file_content)

all_sections = texsoup.find_all("section")
#file_content[all_sections[0].position: all_sections[1].position]
sorted_sections = list(sorted(all_sections, key= lambda s: s.position))
bibliography = texsoup.find(bibliography_block_name)

sections_parsed = {};
for i, section in enumerate(sorted_sections):
    section_text_content = ''
    if i == (len(sorted_sections)-1):
        section_text_content = file_content[section.position: bibliography.position]
    else:
        section_text_content = file_content[section.position: sorted_sections[i+1].position]
    sections_parsed[str(section.string).lower()] = section_text_content

sections_parsed_trimmed = {}
for i, (key, value) in enumerate(sections_parsed.items()):
    parsed_section = TexSoup(value)
    # items_to_be_merged_to_string = Enumerable(
    #     parsed_section.all[1:]).select(
    #     lambda item: str(item.string)).where(
    #     lambda item: item != '' and item != 'None')

    section_text_items = parsed_section.all[1:]
    items_to_be_merged_to_string = list(
        filter(lambda item: item.strip() != '',
               map(lambda item: str(item), section_text_items)))
    final_string = ''.join(items_to_be_merged_to_string)
    sections_parsed_trimmed[key] = final_string

#bibliography = extract_bibliography(texsoup)
standalone_extracts, errors = extract_standalone(texsoup)

print(list(texsoup.find_all('abstract')))
######
# Путь к папке с файлами
tex_folder = "G:/UserData/Desktop/disser/TeX-collection-sections-fix/TeX-collection"
file_paths = [os.path.join(tex_folder, f) for f in os.listdir(tex_folder) if f.endswith(".tex")]

data = []
labels = []
error_files = []

for path in file_paths:
    try:
        try:
            with open(path, "r", encoding="utf-8") as f:
                tex_content = f.read()
        except UnicodeDecodeError:
            try:
                with open(path, "r", encoding="latin-1") as f:
                    tex_content = f.read()
            except UnicodeDecodeError:
                with open(path, "r", encoding="cp1252") as f:
                    tex_content = f.read()

        sections = extract_sections_lib(tex_content)
        for title, content in sections:
            data.append(content)
            labels.append(title)
    except Exception as e:
        print(f"Ошибка при обработке файла {path}: {e}")
        error_files.append(path)

# TF-IDF векторизация
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data).toarray()

# Кодируем метки
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Разбиваем на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Конвертируем в тензоры
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Создаём DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


# Модель на PyTorch
class DocumentBlockClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(DocumentBlockClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Инициализируем модель
num_classes = len(label_encoder.classes_)
model = DocumentBlockClassifier(input_dim=X.shape[1], num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Обучение модели
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")


train_model(model, train_loader, criterion, optimizer)


# Оценка качества модели
def evaluate(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    print(f"Accuracy: {100 * correct / total:.2f}%")


evaluate(model, test_loader)
