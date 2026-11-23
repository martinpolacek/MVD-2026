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

## Úloha 1 - ELECTRA na angličtině

1. Naprogramujte trénovací smyčku a funkci `evaluate(model, texty, popisky)`, která vrátí přesnost a F1. Texty tokenizujte po dávkách (`tok(texty, padding=True, truncation=True, max_length=512, return_tensors="pt")`) a modelu předejte `input_ids` a `attention_mask`. Model vytvořte pomocí `AutoModelForSequenceClassification.from_pretrained(..., num_labels=2)`.

2. Dolaďte mELECTRA na 1 000 anglických recenzích. Vyzkoušejte aspoň tři learning raty a vykreslete trénovací loss. Při příliš velkém learning ratu se model zhroutí (loss zůstane kolem 0,69) a předpovídá jen jednu třídu. Tak poznáte, že jste to přehnali.

3. Vyhodnoťte model na českém `dev` a `test`. Česky se přitom vůbec neučil.

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
