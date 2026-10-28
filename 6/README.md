# Cvičení 6 - PageRank, HITS a vektory webových stránek

Úloha 1: PageRank a HITS. Úloha 2: kategorie webu z vektorů modelu HTML-LM (MetaCentrum, GPU) a srovnání s TF-IDF.

Potřebné knihovny: `numpy`, `scikit-learn`, `beautifulsoup4`, `lxml`, `torch`, `transformers`

## Úloha 1 - PageRank a HITS

Data: `graph_1.txt` až `graph_6.txt` z [tohoto repozitáře](https://github.com/chonyy/PageRank-HITS-SimRank/tree/master/dataset). Řádek = odkaz `odkud,kam`, uzly od 1. Jen `numpy`, vektorizovaně (žádné cykly přes uzly), implementaci z repozitáře nepoužívejte.

1. Matice sousednosti A (N × N): A<sub>ij</sub> = 1, pokud i odkazuje na j.
2. **PageRank**, α = 0,2, 100 iterací. M = A s řádky vydělenými počtem odkazů, E = matice jedniček. Stránka bez odkazů má α = 1.

```math
p^{(0)} = \left(\tfrac{1}{N}, \dots, \tfrac{1}{N}\right)^T, \qquad A' = (1-\alpha)\,M + \frac{\alpha}{N} E, \qquad p^{(i+1)} = A'^T p^{(i)}
```

3. **HITS**, 100 iterací, po každé iteraci oba vektory vydělte jejich součtem:

```math
a^{(0)} = h^{(0)} = (1, \dots, 1)^T, \qquad h^{(i+1)} = A\,a^{(i)}, \qquad a^{(i+1)} = A^T h^{(i)}
```

4. Po kolika iteracích se PageRank přestane měnit (součet |p<sup>(i+1)</sup> − p<sup>(i)</sup>| < 10<sup>−10</sup>)? Stačí 100 iterací i pro grafy 5 a 6?
5. Graf 6: 5 uzlů s nejvyšším PageRankem, 5 nejlepších autorit a hubů. Porovnejte s počtem odkazů na uzel. Proč se PageRank a autorita liší?

Kontrola (uzly 1, 2, …):

| Graf | PageRank | Autorita | Hub |
|---|---|---|---|
| 1 | 0,0656 0,1181 0,1601 0,1937 0,2205 0,2420 | 0 0,2 0,2 0,2 0,2 0,2 | 0,2 0,2 0,2 0,2 0,2 0 |
| 3 | 0,1786 0,3214 0,3214 0,1786 | 0,1910 0,3090 0,3090 0,1910 | 0,1910 0,3090 0,3090 0,1910 |
| 4 | 0,2726 0,1567 0,1384 0,1092 0,1853 0,0656 0,0722 | 0,1395 0,1779 0,2008 0,1402 0,2014 0,0561 0,0841 | 0,2755 0,0478 0,1087 0,1987 0,1837 0,1167 0,0690 |

## Úloha 2 - Kategorie webu pomocí HTML-LM na MetaCentru

[HTML-LM](https://huggingface.co/Seznam/html-lm) (Seznam.cz) převede HTML stránku na vektor délky 768. Model se neučí, jen převádí. Učí se pouze logistická regrese nad vektory.

Data: [`curlie_cs.jsonl.gz`](https://github.com/martinpolacek/MVD-2026/releases/download/cv06-data/curlie_cs.jsonl.gz) (134 MB) – 11 116 úvodních stránek českých webů, 14 kategorií z katalogu [Curlie](https://curlie.org) (dataset [Homepage2Vec](https://doi.org/10.6084/m9.figshare.19406693), CC BY 4.0). Na MetaCentru: `wget <odkaz>`.

```python
import gzip, json
pages = [json.loads(line) for line in gzip.open("curlie_cs.jsonl.gz", "rt", encoding="utf-8")]
# klíče: uid, url, category, split (train / test), html
```

Vektory (stránky v dávce seřaďte podle délky):

```python
import torch
from transformers import AutoModel, AutoProcessor

processor = AutoProcessor.from_pretrained("Seznam/html-lm", trust_remote_code=True)
model = AutoModel.from_pretrained("Seznam/html-lm").to("cuda").eval()

ids = processor(html_texty, truncation=True, max_length=4096)["input_ids"]
x = processor.tokenizer.pad({"input_ids": ids[:16]}, return_tensors="pt").to("cuda")
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    out = model(input_ids=x["input_ids"], attention_mask=x["attention_mask"]).last_hidden_state
vektory = out[:, 0].float().cpu().numpy()      # vektor [CLS], rozměr (16, 768)
```

Klasifikátor:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

1. Skript, který převede stránky na vektory a uloží `.npy`. Spusťte na MetaCentru s GPU (fronta `gpu`, `ngpus=1`, stačí 8 GB). Trvá několik minut.
2. Logistická regrese nad vektory: učení na `split == "train"`, test na `split == "test"`. Bez `StandardScaler` to funguje špatně.
3. **Totéž bez HTML-LM**, ať víte, jestli model pomáhá: z HTML vezměte jen text, převeďte ho na TF-IDF vektory (jako ve 4. cvičení) a naučte nad nimi stejnou logistickou regresi. TF-IDF vektory jsou řídké a už normalizované, `StandardScaler` se nepoužije:

```python
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

texty = [BeautifulSoup(p["html"], "lxml").get_text(" ") for p in pages]
tfidf = TfidfVectorizer(min_df=2, sublinear_tf=True).fit(texty_train)        # slovník jen z učicích stránek
clf_tfidf = LogisticRegression(max_iter=3000, C=10).fit(tfidf.transform(texty_train), y_train)
y_pred_tfidf = clf_tfidf.predict(tfidf.transform(texty_test))
```
4. U obou metod přesnost, macro-F1 a `classification_report`:
    - přesnost = podíl správně zařazených testovacích stránek,
    - F1 kategorie = harmonický průměr precision a recall (přednáška 4),
    - macro-F1 = průměr F1 přes 14 kategorií, každá má stejnou váhu (`f1_score(..., average="macro")`), takže malé kategorie (News, Games) se počítají stejně jako velký Business,
    - `classification_report(y_test, y_pred)` (ze `sklearn.metrics`) vypíše tabulku: pro každou kategorii precision, recall, F1 a support (počet testovacích stránek kategorie), pod ní accuracy a řádek `macro avg` (jeho F1 je macro-F1).

   Které kategorie jdou dobře a které špatně? Je TF-IDF horší než HTML-LM, a proč? (Seznam uvádí pro HTML-LM přesnost 0,76.)
5. **Pročištění HTML:** před processorem odstraňte menu, hlavičku, patičku a formuláře (`nav`, `header`, `footer`, `aside`, `form`…), případně nechte jen `<main>` / `<article>`. Spočítejte vektory znovu (MetaCentrum) a porovnejte přesnost. Kolik stránek se vejde do 4096 tokenů před a po? Pomohlo to, nebo uškodilo, a proč?

Kontrola: HTML-LM i TF-IDF na neupraveném HTML mají přesnost kolem 0,75.

## Odevzdání

Složka `cv06` ve vašem repozitáři:
- notebook s úlohou 1 a klasifikací z úlohy 2,
- skript pro vektory a PBS skript, logy z MetaCentra,
- tabulka přesnosti a macro-F1: HTML-LM (neupravené i pročištěné HTML) a TF-IDF, odpovědi na otázky.

Data ani vektory do repozitáře nenahrávejte.
