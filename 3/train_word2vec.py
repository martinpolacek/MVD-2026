import collections
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL = sys.argv[1] if len(sys.argv) > 1 else "skipgram"   # "skipgram" nebo "cbow"
DIM = 300          # dimenze vektorů
WINDOW = 5         # počet slov vlevo i vpravo od středového slova
NEGATIVES = 5      # počet náhodných slov ke každému správnému
EPOCHS = 5
BATCH_SIZE = 4096
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Data (hotové) ----------
words = open("text8").read().split()
counts = collections.Counter(words)
vocab = [w for w, c in counts.most_common() if c >= 5]
word2idx = {w: i for i, w in enumerate(vocab)}
freq = torch.tensor([counts[w] for w in vocab], dtype=torch.float) / len(words)

ids = torch.tensor([word2idx[w] for w in words if w in word2idx])
ids = ids[torch.rand(len(ids)) < (1e-4 / freq[ids]).sqrt() + 1e-4 / freq[ids]]   # vynechání častých slov (the, of, ...)

centers = ids[WINDOW:-WINDOW]                                       # (N,)
offsets = [o for o in range(-WINDOW, WINDOW + 1) if o != 0]
windows = torch.stack([ids[WINDOW + o: len(ids) - WINDOW + o] for o in offsets], dim=1)   # (N, 2*WINDOW)

if MODEL == "skipgram":   # vstup: středové slovo, cíl: jedno slovo z okna
    inputs, targets = centers.repeat_interleave(2 * WINDOW), windows.flatten()
else:                     # vstup: celé okno, cíl: středové slovo
    inputs, targets = windows, centers
inputs, targets = inputs.to(device), targets.to(device)
noise = (freq ** 0.75).to(device)                                  # pravděpodobnosti pro náhodná slova
print(f"slovník: {len(vocab):,} slov, trénovacích příkladů: {len(targets):,}, zařízení: {device}", flush=True)


def batches():
    """Zamíchá data a vrací dávky (vstup, kandidáti, labely).

    Ke každému vstupu je 1 + NEGATIVES kandidátů: na pozici 0 správné slovo (label 1),
    za ním náhodná slova (label 0). Model se tedy učí rozlišit správné slovo od náhodných.
    """
    perm = torch.randperm(len(targets), device=device)
    for i in range(0, len(targets), BATCH_SIZE):
        idx = perm[i: i + BATCH_SIZE]
        negatives = torch.multinomial(noise, len(idx) * NEGATIVES, replacement=True).view(-1, NEGATIVES)
        candidates = torch.cat([targets[idx].unsqueeze(1), negatives], dim=1)
        labels = torch.zeros(candidates.shape, device=device)
        labels[:, 0] = 1
        yield inputs[idx], candidates, labels


def save_vectors(model):
    """Uloží vektory slov ve stejném formátu jako GloVe."""
    with open(f"vectors_{MODEL}.txt", "w") as f:
        for word, vector in zip(vocab, model.in_emb.weight.tolist()):
            f.write(word + " " + " ".join(f"{x:.5f}" for x in vector) + "\n")


# ---------- Model ----------
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.in_emb = nn.Embedding(vocab_size, dim, sparse=True)    # vektory vstupních slov (výsledek)
        self.out_emb = nn.Embedding(vocab_size, dim, sparse=True)   # vektory kandidátů
        nn.init.uniform_(self.in_emb.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.out_emb.weight)

    def forward(self, x, candidates):
        # TODO:
        #   1. vektor vstupního slova v (self.in_emb), u CBOW průměr vektorů slov z okna
        #   2. vektory kandidátů u (self.out_emb)
        #   3. vraťte skóre = skalární součin v s každým kandidátem, tvar (B, 1 + NEGATIVES)
        raise NotImplementedError


# ---------- Trénink ----------
model = Word2Vec(len(vocab), DIM).to(device)
optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.003)

start = time.time()
for epoch in range(EPOCHS):
    for x, candidates, labels in batches():
        # TODO: jeden krok učení
        #   1. scores = skóre kandidátů z modelu
        #   2. loss = F.binary_cross_entropy_with_logits(scores, labels)
        #   3. vynulujte gradienty, spočítejte gradienty (backward) a udělejte krok optimizeru
        raise NotImplementedError
    print(f"epocha {epoch + 1}: loss {loss.item():.4f}, čas {time.time() - start:.0f} s", flush=True)

save_vectors(model)
