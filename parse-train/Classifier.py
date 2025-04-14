import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

def prepare_dataset_from_json(folder_path):
    texts = []
    labels = []
    error_files = []

    for fname in os.listdir(folder_path):
        if not fname.endswith("tex_sections.json"):
            continue
        fpath = os.path.join(folder_path, fname)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                sections = data.get("sections", {})
                for label, content in sections.items():
                    if not content:
                        continue
                    if label == "main_part" or label == "main_parts":
                        # Если это список, добавляем каждый элемент отдельно
                        if isinstance(content, list):
                            for part in content:
                                if part.strip():
                                    texts.append(part.strip())
                                    labels.append(label)
                        else:
                            # иногда main_part может быть строкой по ошибке
                            if content.strip():
                                texts.append(content)
                                labels.append(label)
                    else:
                        if isinstance(content, str) and content.strip():
                            texts.append(content)
                            labels.append(label)
        except Exception as e:
            print(f"Ошибка при обработке {fname}: {e}")
            error_files.append(fname)

    return texts, labels, error_files

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train_model(model, train_loader, criterion, optimizer, metrics, epochs=10):
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        metrics[f"epoch_{epoch+1}_loss"] = avg_loss

# Обучение модели
def evaluate(model, test_loader, metrics):
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    metrics["accuracy"] = accuracy

# Функция для загрузки модели
def load_model(model_path, input_dim, num_classes):
    model = Classifier(input_dim, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def train(json_folder='processed_tex'):
    epochs = 10

    texts, labels, error_files = prepare_dataset_from_json(json_folder)

    if not texts or not labels:
        raise RuntimeError("Нет валидных данных для обучения. Проверьте json-файлы.")

    # TF-IDF векторизация
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts).toarray()

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

    # Инициализируем модель
    num_classes = len(label_encoder.classes_)
    model = Classifier(input_dim=X.shape[1], num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Обучение модели
    metrics = {}

    train_model(model, train_loader, criterion, optimizer, metrics, epochs)
    evaluate(model, test_loader, metrics)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join("models", timestamp)
    os.makedirs(model_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(model_dir, "classifier_model.pt"))
    with open(os.path.join(model_dir, "training_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(vectorizer, os.path.join(model_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.pkl"))


train()