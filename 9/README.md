# Cvičení 9 - BERT: maskování a kontextové embeddingy

Úloha 1: tokenizace a predikce maskovaného slova. Úloha 2: kontextové embeddingy slova „bank“ ve vrstvách BERT. Bonus: automatické nalezení významů slova ve Wikipedii.

Model: [`bert-base-uncased`](https://huggingface.co/bert-base-uncased) z knihovny `transformers`. Vše běží na CPU (bonus asi 1–2 minuty).

Potřebné knihovny: `transformers`, `torch`, `numpy`, `matplotlib`, `scikit-learn`, pro bonus `datasets`

```python
from transformers import AutoTokenizer, AutoModel, pipeline

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased").eval()
```

## Úloha 1 - Tokenizace a maskované slovo

1. Vypište tokeny (`tok.convert_ids_to_tokens(tok(veta)["input_ids"])`) a jejich počet pro věty `"Hello, this is Bert."` a `"Tokenization of unbelievably long words."`. U každé věty určete, které tokeny přidal tokenizer navíc a které slovo se rozdělilo na více tokenů (začínají `##`).

2. Pomocí `pipeline("fill-mask", model="bert-base-uncased")` vypište 5 nejpravděpodobnějších slov a jejich pravděpodobnost pro věty:
   - `The cat sat on the [MASK].`
   - `Paris is the capital of [MASK].`
   - `I went to the [MASK] to withdraw some money.`
   - `The man worked as a [MASK].`
   - `The woman worked as a [MASK].`

Kontrola:

| Věta | Tokenů | Nejpravděpodobnější slovo |
|---|---|---|
| Hello, this is Bert. | 8 | – |
| Tokenization of unbelievably long words. | 13 | – |
| The cat sat on the [MASK]. | | floor (0,31) |
| Paris is the capital of [MASK]. | | france (0,95) |
| I went to the [MASK] to withdraw some money. | | bank (0,96) |
| The man / woman worked as a [MASK]. | | carpenter (0,10) / nurse (0,22) |

## Úloha 2 - Kontextové embeddingy

Slovo „bank“ znamená břeh řeky i banku. Statický embedding (Word2Vec, GloVe) má pro obě situace stejný vektor, BERT dá v každé větě jiný.

```python
vety = ["We sat on the bank of the river.", "The boat drifted to the muddy bank.",
        "Fishermen stood on the river bank.",                       # břeh
        "I deposited money at the bank.", "The bank approved my loan.",
        "She works at a bank in Prague."]                           # banka
```

1. Pro každou větu spusťte model s `output_hidden_states=True`. Výstup `hidden_states` obsahuje 13 matic: vrstvu 0 (vstupní embeddingy) a výstupy 12 vrstev kodéru. Z každé vezměte vektor na pozici tokenu `bank` (pozici najděte podle `tok.convert_tokens_to_ids("bank")`, ne podle pořadí slova ve větě).

2. Pro poslední vrstvu spočítejte a vypište matici kosinových podobností 6 × 6.

3. Pro každou vrstvu 0–12 spočítejte průměrnou kosinovou podobnost dvojic **se stejným významem** (bez podobnosti vektoru se sebou samým) a dvojic **s různým významem**. Obě křivky vykreslete do jednoho grafu (osa x = vrstva). Ukázka: [`sample_vrstvy.png`](sample_vrstvy.png).

Kontrola:

| Vrstva | Stejný význam | Různý význam |
|---|---|---|
| 0 | 0,96 | 0,97 |
| 6 | 0,80 | 0,52 |
| 12 | 0,78 | 0,46 |

V poslední vrstvě je podobnost dvojic „břeh–břeh“ 0,79–0,81 a dvojic „břeh–banka“ 0,41–0,54.

## Bonus - Významy slova „bank“ ve Wikipedii

Ve Wikipedii nikdo neoznačil, ve kterém významu je slovo použité. Zkuste významy najít automaticky shlukováním kontextových embeddingů.

1. Ze Simple Wikipedie vyberte všechny věty se slovem „bank“:

```python
import re
from datasets import load_dataset

wiki = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train")
vety = set()
for text in wiki["text"]:
    if "bank" not in text.lower():
        continue
    for s in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")):
        if re.search(r"\bbank\b", s.lower()) and 6 <= len(s.split()) <= 40:
            vety.add(s.strip())
vety = sorted(vety)
```

2. Pro každou větu spočítejte embedding tokenu `bank` jako **průměr posledních 4 vrstev** (`hidden_states[-4:]`). Věty zpracovávejte po dávkách (např. 64 vět, `padding=True`, `truncation=True, max_length=128`) a bez gradientu (`torch.no_grad()`). Věty, ve kterých tokenizer slovo `bank` nemá jako samostatný token, vynechte.

3. Embeddingy rozdělte vaším K-means ze 7. cvičení do **K = 3** shluků (10 startů, nejmenší J). Pro každý shluk vypište počet vět a 8 typických slov. Typická slova spočítejte stejně jako v 7. cvičení, jen místo TF-IDF stačí `CountVectorizer(stop_words="english", min_df=5, binary=True)` nad větami. Podle slov shluky pojmenujte.

4. Embeddingy promítněte do 2D (PCA nebo t-SNE) a vykreslete obarvené podle shluků. Ukázka: [`sample_bank_mapa.png`](sample_bank_mapa.png); barvy a pořadí shluků se u vás mohou lišit.

5. Pro nové věty spočítejte embedding stejně jako v kroku 2, přiřaďte je k nejbližšímu středu shluku a vypište vzdálenosti ke všem třem středům:
   - `We had a picnic on the bank of the lake.`
   - `The bank raised interest rates again.`
   - `Settlers moved into the West Bank.`
   - `The plane had to bank sharply to the left.`

   Poslední věta používá „bank“ jako sloveso (naklonit se). Do kterého shluku spadne a jak daleko je od nejbližšího středu ve srovnání s ostatními větami?

Kontrola:
- vět se samostatným tokenem `bank`: 3410,
- shluky asi 2640 vět (banka: money, central, president, …), asi 585 vět (břeh: river, north, left, …) a asi 185 vět (West Bank: israel, gaza, palestinian, …),
- první tři nové věty spadnou do „svých“ shluků se vzdáleností k nejbližšímu středu 7–12, věta s letadlem má od nejbližšího středu asi 18.

## Odevzdání

Složka `cv09` ve vašem repozitáři: notebook s úlohami 1 a 2 (a bonusem) včetně výstupů a grafů.
