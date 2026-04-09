import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for label, text in data_iter:
        yield tokenizer(text)

train_iter = IMDB(split='train')
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])


def text_pipeline(x):
    return vocab(tokenizer(x))

def label_pipeline(x):
    return 1 if x == 'pos' else 0

def collate_batch(batch):
    labels, texts = [], []
    for label, text in batch:
        labels.append(label_pipeline(label))
        processed = text_pipeline(text)[:200]
        texts.append(torch.tensor(processed))
    texts = nn.utils.rnn.pad_sequence(texts, batch_first=True)
    labels = torch.tensor(labels)
    return texts, labels

train_iter, test_iter = IMDB()
train_loader = DataLoader(list(train_iter), batch_size=32, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(list(test_iter), batch_size=32, collate_fn=collate_batch)

class CNNModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128)
        self.conv = nn.Conv1d(128, 128, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = self.embedding(x).permute(0,2,1)
        x = self.pool(self.conv(x))
        x = torch.mean(x, dim=2)
        return self.fc(x)


class RNNModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128)
        self.lstm = nn.LSTM(128, 64, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        return self.fc(x[:, -1, :])


class CNN_RNN_Model(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128)
        self.conv = nn.Conv1d(128, 128, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(128, 64, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.embedding(x).permute(0,2,1)
        x = self.pool(self.conv(x)).permute(0,2,1)
        x, _ = self.lstm(x)
        return self.fc(x[:, -1, :])


class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        return x * torch.sigmoid(self.fc(x))


class Attention_CNN_RNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128)
        self.conv = nn.Conv1d(128, 128, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.attn = Attention(128)
        self.lstm = nn.LSTM(128, 64, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.embedding(x).permute(0,2,1)
        x = self.pool(self.conv(x)).permute(0,2,1)
        x = self.attn(x)
        x, _ = self.lstm(x)
        return self.fc(x[:, -1, :])

# -------------------------
# Training
# -------------------------
def train(model, loader):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(3):
        total_loss = 0
        for texts, labels in loader:
            texts, labels = texts.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for texts, labels in loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            preds = outputs.argmax(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return acc, prec, rec, f1


models = {
    "CNN": CNNModel(len(vocab)),
    "RNN": RNNModel(len(vocab)),
    "CNN-RNN": CNN_RNN_Model(len(vocab)),
    "Attention CNN-RNN": Attention_CNN_RNN(len(vocab))
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model = model.to(device)

    train(model, train_loader)
    acc, prec, rec, f1 = evaluate(model, test_loader)

    results[name] = (acc, prec, rec, f1)

print("\nFINAL RESULTS")
for name, (acc, prec, rec, f1) in results.items():
    print(f"{name}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
