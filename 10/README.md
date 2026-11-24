# Cvičení 10 - Sentiment českých recenzí bez českých popisků

Úloha 1: ELECTRA na angličtině. Úloha 2: větší model jako učitel. Úloha 3: ELECTRA na slabých popiscích. Bonus: co nejlepší ELECTRA.

Máte popsané jen anglické recenze, ale klasifikátor potřebujete na české. Větší vícejazyčný model se naučí úlohu z angličtiny, **označí české recenze** (slabé popisky, weak labels) a malá ELECTRA se naučí z nich.

- malý model: [`AILabTUL/mELECTRA`](https://huggingface.co/AILabTUL/mELECTRA) (13,5 M parametrů)
- učitel: [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) (118 M parametrů)

Trénování pouštějte na GPU (MetaCentrum nebo Google Colab).

Potřebné knihovny: `transformers`, `torch`, `datasets`, `scikit-learn`, `matplotlib`

## Data

```python
from datasets import load_dataset

# 1 000 anglických recenzí s popiskem (0 = negativní, 1 = pozitivní)
imdb = load_dataset("stanfordnlp/imdb", split="train").shuffle(seed=42).select(range(1000))
en_text, en_label = list(imdb["text"]), list(imdb["label"])

# 5 000 českých recenzí BEZ popisku
csfd = load_dataset("fewshot-goes-multilingual/cs_csfd-movie-reviews")
cs_unlabeled = list(csfd["train"].shuffle(seed=42).select(range(5000))["comment"])

# české recenze s popiskem, jen na vyhodnocení (0-1 hvězda = negativní, 4-5 = pozitivní)
def binarni(split):
    d = csfd[split].filter(lambda r: r["rating_int"] in (0, 1, 4, 5))
    return list(d["comment"]), [int(r >= 4) for r in d["rating_int"]]

dev_text, dev_label = binarni("validation")
dev_text, dev_label = dev_text[:500], dev_label[:500]
test_text, test_label = binarni("test")
```

Pravidla:
- Trénujte jen na `en_text` s `en_label` a na `cs_unlabeled`. Žádná další data, žádné hvězdičky z ČSFD.
- `dev` slouží k ladění (learning rate, počet epoch), `test` jen k závěrečnému vyhodnocení. Na žádném z nich netrénujte.
- Všechny výsledky uvádějte jako **průměr F1 ze tří seedů** (a vypište i jednotlivé hodnoty). F1 počítejte přes `f1_score(y, p, average="weighted")`.

## Jak na to

Ukázky níže jsou stavební kameny, smyčku přes epochy, vyhodnocení a ukládání výsledků si z nich složíte sami. Stejný kód použijete pro oba modely, liší se jen jméno modelu, data a parametry trénování.

**1. Tokenizer a model.** Obojí se podle jména samo stáhne z Hugging Face.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = "cuda" if torch.cuda.is_available() else "cpu"
name = "AILabTUL/mELECTRA"                       # nebo "intfloat/multilingual-e5-small"

tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).to(device)
```

`AutoModelForSequenceClassification` přidá nad předtrénovaný kodér klasifikační hlavu se dvěma výstupy. Hlava je na začátku náhodná, proto knihovna vypíše varování, že některé váhy nejsou inicializované. To je v pořádku.

**2. Z textu na čísla.** Recenze v dávce jsou různě dlouhé, tokenizer je doplní na stejnou délku (`padding`) a příliš dlouhé ořízne (`truncation`).

```python
enc = tok(["Great movie!", "Nejhorší film, jaký jsem kdy viděl."],
          padding=True, truncation=True, max_length=512, return_tensors="pt")
print(enc["input_ids"].shape)                    # (počet textů, délka nejdelšího textu)
print(enc["attention_mask"])                     # 1 = skutečný token, 0 = výplň
```

**3. Dávky.** `DataLoader` vybírá náhodné dávky a funkce `collate` každou z nich tokenizuje:

```python
from torch.utils.data import DataLoader

def collate(batch):
    texts, labels = zip(*batch)
    enc = tok(list(texts), padding=True, truncation=True, max_length=512, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"], torch.tensor(labels)

loader = DataLoader(list(zip(en_text, en_label)), batch_size=16, shuffle=True, collate_fn=collate)
```

**4. Jeden trénovací krok.** Když modelu předáte `labels`, vrátí rovnou i loss (křížovou entropii). Learning rate během prvních 10 % kroků roste (warmup) a pak lineárně klesá k nule:

```python
from transformers import get_linear_schedule_with_warmup

epochs, lr = 10, 5e-5
steps = len(loader) * epochs
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)

model.train()
ids, mask, y = next(iter(loader))                # jedna dávka, ve smyčce projdete všechny
out = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=y.to(device))
out.loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
scheduler.step()
optimizer.zero_grad()
print(out.loss.item())                           # na začátku kolem 0,69 = ln 2, model hádá
```

**5. Předpověď.** Při vyhodnocení přepněte model do `eval()` (vypne dropout) a vypněte gradienty. Výstup `logits` má tvar (počet textů, 2), třída je sloupec s větší hodnotou:

```python
model.eval()
with torch.no_grad():
    enc = tok(dev_text[:64], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
pred = out.logits.argmax(dim=1).cpu().tolist()   # 0 / 1 pro každý text
```

Takhle zpracujete jednu dávku 64 textů. Pro všechny texty projděte dávky ve smyčce **bez zamíchání**, jinak předpovědi nebudou odpovídat popiskům. Stejně získáte i slabé popisky v úloze 2.

**6. Seedy.** Před vytvořením modelu nastavte `torch.manual_seed(seed)`. Pro další seed vytvořte model znovu, jinak byste pokračovali v už natrénovaném. Starý model předtím uvolněte z paměti GPU: `del model; torch.cuda.empty_cache()`.

Doporučené parametry na začátek (`max_length=512`, warmup 10 %):

| Trénování | batch | epoch | learning rate |
|---|---|---|---|
| mELECTRA, úloha 1 | 16 | 10 | vyzkoušejte sami, např. 2e-5, 1e-4, 3e-4 |
| e5-small, úloha 2 | 16 | 5 | 5e-5 |
| mELECTRA, úloha 3 | 32 | 8 | podle úlohy 1 |

Kód si nejdřív vyzkoušejte na pár dávkách a teprve pak ho pusťte celý na MetaCentru jako dávkovou úlohu s GPU (fronta `gpu`, `ngpus=1`, stejně jako ve cvičeních 3 a 5).

Pro představu: na kartě NVIDIA L40 trvá úloha 1 i učitel asi 40 sekund na jeden seed a stačí 8 GB paměti GPU.

## Úloha 1 - ELECTRA na angličtině

1. Podle ukázek naprogramujte funkci `train(name, texty, popisky, epochs, lr, batch_size, seed)`, která vrátí natrénovaný model s tokenizerem, a funkci `evaluate(tok, model, texty, popisky)`, která vrátí přesnost a F1 (`accuracy_score` a `f1_score` ze `sklearn.metrics`). Budete je používat ve všech úlohách.

2. Dolaďte mELECTRA na 1 000 anglických recenzích. Vyzkoušejte aspoň tři learning raty, vykreslete průměrnou trénovací loss po epochách a nejlepší learning rate vyberte podle `dev`. Při příliš velkém learning ratu se model zhroutí: loss zůstane kolem 0,69 a model předpovídá pořád stejnou třídu.

3. Vybraný model vyhodnoťte na českém `dev` a `test`. Na českých recenzích se přitom nedolaďoval, češtinu zná jen z předtrénování.

## Úloha 2 - Učitel

1. Stejným kódem dolaďte `multilingual-e5-small` na stejných 1 000 anglických recenzích a vyhodnoťte ho na českém `dev` a `test`.

2. Učitelem označte všech 5 000 recenzí z `cs_unlabeled`. Vypište podíl pozitivních popisků. Pokud vyjde blízko 0 nebo 1, učitel češtinu nezvládá a popisky jsou k ničemu.

## Úloha 3 - ELECTRA na slabých popiscích

1. Natrénujte **novou** mELECTRA na 5 000 českých recenzích s popisky od učitele a vyhodnoťte ji na `dev` a `test`. Na `test` má dosáhnout průměrné **F1 aspoň 72 %**.

2. Do jedné tabulky dejte ELECTRA z úlohy 1, učitele a ELECTRA z úlohy 3: počet parametrů, F1 na `dev` a na `test`.

## Bonus

Jeden bod získá, kdo s **mELECTRA** dosáhne na `test` průměrné **F1 aspoň 82 %**. Platí stejná pravidla pro data jako výše; postup je na vás. Předtrénované modely z Hugging Face jsou povolené, výsledný klasifikátor ale musí být mELECTRA.

## Kontrola

| Model | F1 na českém `test` |
|---|---|
| mELECTRA po úloze 1 | asi 63 % |
| učitel e5-small | asi 76 % |
| mELECTRA po úloze 3 | asi 76 % |

## Odevzdání

Složka `cv10` ve vašem repozitáři: notebook s úlohami 1–3 (a bonusem) včetně výstupů, tabulky a grafu loss.
