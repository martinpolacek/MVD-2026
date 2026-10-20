# Cvičení 5 - Vyhledávání pomocí vektorů

Cílem cvičení je převést dotaz i dokument na vektor a relevanci počítat jako kosinovou podobnost, místo počítání společných slov jako v TF-IDF. V úloze 1 sestavíte vektor dokumentu z GloVe vektorů slov. V úloze 2 použijete předtrénovanou neuronovou síť (siamská architektura z přednášky), na MetaCentru s ní převedete na vektory celou Simple Wikipedii a porovnáte výsledky s TF-IDF indexem ze 4. cvičení.

Potřebné knihovny: `numpy`, `pandas`, `spacy`, `scikit-learn`, `datasets`, `sentence-transformers`

## Úloha 1 - TF-IDF vážený průměr GloVe vektorů

Použijte články z Medium ze 4. cvičení ([`archive.zip`](../4/archive.zip)) a GloVe vektory ze 3. cvičení (stačí 100d, výsledky spusťte i s 300d).

1. Načtěte články. Titulek a text spojte do jednoho dokumentu a předzpracujte stejnou funkcí jako ve 4. cvičení. Dataset obsahuje některé články vícekrát (337 řádků, ale jen 230 různých článků), duplicity odstraňte (`drop_duplicates`).
2. Spočítejte TF-IDF matici pomocí [`TfidfVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) ze `sklearn`. Text už je předzpracovaný, nastavte proto, aby jen dělil podle mezer:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False)
X = vectorizer.fit_transform(docs)           # řídká matice (počet dokumentů, velikost slovníku)
vocab = vectorizer.get_feature_names_out()   # slovo pro každý sloupec X
```

3. Vektor dokumentu spočítejte jako vážený průměr GloVe vektorů jeho slov, váhou je TF-IDF slova v dokumentu:

```math
doc\_vector(d) = \frac{\sum_{w \in d} \text{TF-IDF}(w, d)\; glove(w)}{\sum_{w \in d} \text{TF-IDF}(w, d)}
```

   Slova, která v GloVe nejsou, vynechte (v čitateli i ve jmenovateli). Kolik procent slov ze slovníku v GloVe chybí a jaká slova to jsou?

   Tip: nemusíte procházet slova v cyklu. Sestavte matici `E` o rozměru *(velikost slovníku, d)*, kde řádek `i` je GloVe vektor slova `vocab[i]` (pro slova mimo GloVe nulový řádek). Čitatel pro všechny dokumenty najednou je pak `X @ E`.

4. Dotaz převeďte na vektor stejným způsobem (`vectorizer.transform`) a dokumenty seřaďte podle kosinové podobnosti:

```math
score(q,d) = \cos(query\_vector, doc\_vector)
```

5. Pro dotaz `coursera vs udacity machine learning` a aspoň dva vlastní dotazy vypište 5 nejlepších článků těmito třemi metodami:
    - TF-IDF ze 4. cvičení,
    - prostý průměr GloVe vektorů slov,
    - TF-IDF vážený průměr GloVe vektorů.

   Výsledky porovnejte. Našel průměr GloVe vektorů článek *Coursera vs Udacity for Machine Learning*? Proč? Jaké hodnoty kosinové podobnosti vycházejí a proč jsou si tak blízké?

## Úloha 2 - Neuronové vyhledávání ve Wikipedii na MetaCentru

Průměr vektorů slov ztrácí pořadí slov i jejich kontext. Lepší vektor dokumentu vytvoří neuronová síť natrénovaná přímo pro vyhledávání. Použijeme model [`multi-qa-MiniLM-L6-cos-v1`](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1). Je to malý BERT (6 vrstev), natrénovaný na 215 milionech párů dotaz-odpověď tak, aby vektor dotazu měl vysokou kosinovou podobnost s vektory relevantních textů. Jde o siamskou (representation-focused) architekturu z přednášky: dotaz i dokument projdou stejnou sítí a relevance je kosinová podobnost jejich vektorů.

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1", device="cuda")
emb = model.encode(texty, batch_size=128, normalize_embeddings=True)   # matice (počet textů, 384)
```

Model přečte nejvýš 512 tokenů, u delších článků tedy jen začátek. Texty si proto můžete předem zkrátit (např. na 2000 znaků), ať se zbytečně netokenizují.

Korpusem je stejná [Simple Wikipedie](https://huggingface.co/datasets/wikimedia/wikipedia/tree/main/20231101.simple) jako ve 4. cvičení (241 787 článků). Do prostředí z 1. cvičení doinstalujte knihovny:

```bash
module add mambaforge
mamba activate /storage/brno2/home/$USER/envs/<vaše_prostředí>
pip install sentence-transformers datasets
```

1. Napište skript, který načte Simple Wikipedii, ke každému článku vytvoří text `titulek. text`, převede všechny články na vektory a uloží je do souboru `.npy`. Uložte je jako `float16` (zabere to asi 185 MB).
2. Skript spusťte na MetaCentru jako dávkovou úlohu s GPU (fronta `gpu`, `ngpus=1`). Model a data se při prvním spuštění stáhnou do `~/.cache/huggingface`. Na GPU trvá převod celého korpusu jen několik minut.
3. Uložené vektory načtěte v notebooku a vyhledávejte: dotaz převeďte na vektor stejným modelem (`normalize_embeddings=True`), skóre všech článků je pak jediné maticové násobení `emb @ q`.
4. Porovnejte výsledky (top 5) s TF-IDF indexem ze 4. cvičení na aspoň 5 dotazech. Zvolte dotazy obou typů:
    - dotazy, které nesdílejí slova s hledaným článkem, např. `how do plants make food from sunlight` (hledáme *Photosynthesis*) nebo `largest animal that ever lived`,
    - dotazy se vzácnými slovy a jmény, např. `coursera vs udacity machine learning`.

   Kdy je lepší který přístup a proč?

## Bonus - Vyhodnocení a našeptávání

Bonus má dvě části. Uznává se, jen když jsou splněné obě.

### Část 1 - Které vyhledávání je nejlepší?

V úloze 2 jste porovnávali výsledky od oka. Teď je změřte na testovací kolekci [SciFact](https://huggingface.co/datasets/BeIR/scifact) z benchmarku BEIR (5 183 abstraktů vědeckých článků, 300 testovacích dotazů a hodnocení relevance):

```python
from datasets import load_dataset
corpus  = load_dataset("BeIR/scifact", "corpus", split="corpus")      # _id, title, text
queries = load_dataset("BeIR/scifact", "queries", split="queries")    # _id, text
qrels   = load_dataset("BeIR/scifact-qrels", split="test")            # query-id, corpus-id, score
```

Vyhodnoťte tři metody: TF-IDF, BM25 (k = 1,2, b = 0,75) a neuronovou síť z úlohy 2. Pro každou spočítejte průměrné **NDCG@10** a **Recall@100** přes všechny testovací dotazy. Metriky si naprogramujte sami podle 4. přednášky (ideální pořadí berte ze všech relevantních dokumentů v qrels). Výsledky shrňte do tabulky. Dopadlo pořadí metod podle očekávání? Pokud ne, čím to může být?

Kontrola: vaše BM25 by mělo mít NDCG@10 kolem 0,66.

### Část 2 - Našeptávání

Natrénujte na MetaCentru na GPU znakovou rekurentní síť (GRU nebo LSTM), která dokončuje rozepsané slovo v dotazu.

- Data: dotazy z [MS MARCO](https://huggingface.co/datasets/BeIR/msmarco), `load_dataset("BeIR/msmarco", "queries", split="queries")`.
- Test: [`nasept_test.tsv`](nasept_test.tsv), 2000 useknutých dotazů (`what time zone is jam` → `jamaica`).

1. Předzpracování jako v testu: malá písmena, jedna mezera, jen ASCII, 3–60 znaků. Z trénovacích dat vyřaďte testovací dotazy podle `query_id` i podle textu.
2. Síť: na vstupu znaky dotazu, předpovídá další znak (cross-entropy). Návrhy generujte např. beam search, dokud síť nevygeneruje mezeru nebo konec dotazu.
3. Metrika **success@3**: podíl prefixů, kde je správné slovo mezi 3 návrhy.
4. Pokusy (velikost, vrstvy, epochy, learning rate) zapište do tabulky.

**Uznání: success@3 ≥ 78 %.**

Návrhy nejlepší sítě odevzdejte v `nasept_pred.tsv`. Úspěšnost z něj přepočítáme a musí sedět s logem. Bez souboru se bonus neuzná.

- UTF-8, oddělovač tabulátor, hlavička `query_id	prefix	navrh1	navrh2	navrh3`,
- 2000 řádků ve stejném pořadí jako `nasept_test.tsv`,
- návrhy jsou celá slova, od nejlepšího, chybějící nechte prázdné.

```
query_id	prefix	navrh1	navrh2	navrh3
398889	i	is	in	if
1158411	city ba	bank	banks	bakery
...
```

## Odevzdání

Do svého repozitáře nahrajte složku `cv05`, která bude obsahovat:
- Jupyter notebook s úlohou 1 a vyhledáváním nad Wikipedií,
- skript pro převod Wikipedie na vektory a PBS skript,
- výstupní logy úloh z MetaCentra,
- porovnání s TF-IDF na 5 dotazech,
- (volitelně) bonus: tabulku metrik na SciFact, skript a log trénování našeptávání, tabulku pokusů a soubor `nasept_pred.tsv`.

Data, GloVe ani uložené vektory do repozitáře nenahrávejte, jsou příliš velké.
