# Cvičení 3 - Word2Vec

Cílem cvičení je vyzkoušet si práci se slovními vektory (word embeddings) a natrénovat si vlastní word2vec model na GPU na MetaCentru. Úlohu 1 zvládnete na vlastním počítači, úlohu 2 spusťte na MetaCentru.

Potřebné knihovny: `numpy`, `torch`

## Úloha 1 - Předtrénované GloVe vektory

1. Stáhněte předtrénované [GloVe vektory](https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip) (Wikipedia 2014 + Gigaword 5, 400 000 slov, 822 MB). Archiv obsahuje vektory o dimenzi 50, 100, 200 a 300. Začněte s nejmenšími, na větších spusťte až hotové řešení.
2. Načtěte data. V textovém souboru je na každém řádku slovo a jeho vektor. Vytvořte:
    - `words` - list slov o délce *n*,
    - `vectors` - matici o rozměru *(n, d)*,
    - `word2idx` - slovník, který vrátí index slova (např. `word2idx["queen"]`).
3. Napište funkci `cossim(a, b)`, která vrátí kosinovou podobnost dvou vektorů:

$$
\text{similarity}(a, b) = \cos(\theta) = \frac{a \cdot b}{\lVert a \rVert \, \lVert b \rVert}
$$

4. Napište funkci pro hledání slovních analogií a vypište 5 nejpodobnějších slov pro analogii `king - man + woman = ?`. Vstupní slova do výsledku nepočítejte. Vyzkoušejte i několik vlastních analogií.

## Úloha 2 - Vlastní word2vec na GPU

Natrénujte vlastní word2vec model (skip-gram) v knihovně PyTorch. Jako korpus slouží [text8](https://mattmahoney.net/dc/textdata.html), tj. prvních 100 MB anglické Wikipedie (17 milionů slov). Stáhněte ho a rozbalte vedle skriptu:

```bash
wget https://mattmahoney.net/dc/text8.zip && unzip text8.zip
```

Kostru máte v souboru [`train_word2vec.py`](train_word2vec.py). Příprava dat je hotová. Trénink funguje jako klasifikace: ke každému vstupnímu slovu dostane model několik kandidátů, jedno správné slovo z jeho okolí (label 1) a několik náhodných slov (label 0), a učí se je rozlišit. Skóre kandidáta je skalární součin jeho vektoru s vektorem vstupního slova.

Doplňte místa označená `TODO`:
- `Word2Vec.forward` - výpočet skóre kandidátů,
- jeden krok učení v tréninkové smyčce.

Než úlohu odešlete, vyzkoušejte si skript lokálně na pár dávkách (např. smyčku po několika krocích ukončete pomocí `break`), ať na MetaCentru nečekáte ve frontě na chybu.

Skript pak spusťte na MetaCentru jako dávkovou úlohu s GPU. V PBS skriptu si kromě paměti a času vyžádejte i grafickou kartu (`ngpus=1`, fronta `gpu`). Pro představu: na CPU běžného notebooku trvá jedna epocha zhruba 20 minut, na GPU desítky sekund.

Výsledné vektory se uloží ve stejném formátu jako GloVe. Načtěte je stejným kódem jako v úloze 1 a porovnejte s GloVe na několika analogiích a nejpodobnějších slovech. Který model je lepší a proč?

## Bonus - CBOW vs. skip-gram

Skip-gram předpovídá z jednoho slova jeho okolí, CBOW naopak z okolí (průměru vektorů slov v okně) předpovídá prostřední slovo. Doplňte do `forward` i variantu pro CBOW a natrénujte oba modely (`python train_word2vec.py skipgram` a `python train_word2vec.py cbow`). Porovnejte je:

1. **Rychlost** - doba trénování jedné epochy.
2. **Přesnost na analogiích** - soubor [`questions-words.txt`](questions-words.txt) obsahuje přes 19 000 analogií rozdělených do sekcí. Spočítejte zvlášť přesnost pro **sémantické** analogie (sekce bez `gram`, např. hlavní města, měny, rodina) a **syntaktické** analogie (sekce `gram*`, např. množné číslo, minulý čas, stupňování).
    - Analogie `a b c d` je správně, pokud je pro `b - a + c` nejpodobnější slovo (kromě `a`, `b`, `c`) právě `d`.
    - Aby bylo porovnání férové (GloVe zná 400 000 slov, váš model asi 71 000), použijte u všech modelů jen 30 000 nejčastějších slov. Soubory s vektory jsou seřazené podle četnosti, stačí tedy načíst prvních 30 000 řádků. Analogie, které obsahují slovo mimo těchto 30 000, přeskočte.
    - Výpočet pro všechny otázky najednou zvládnete maticově (normalizované vektory, maticové násobení, `argmax`).
3. **Nejbližší slova** - vypište 5 nejpodobnějších slov pro několik vybraných slov (např. `king`, `france`, `computer`, `quickly`, `violin`) pro oba modely.

Výsledky shrňte do tabulky a napište, v čem je který model lepší a čím by to mohlo být způsobeno. Pro srovnání stejným způsobem vyhodnoťte i GloVe.

## Odevzdání

Do svého repozitáře nahrajte složku `cv03`, která bude obsahovat:
- Jupyter notebook s úlohou 1 a porovnáním vlastních vektorů s GloVe,
- doplněný `train_word2vec.py` a PBS skript,
- výstupní logy úlohy z MetaCentra,
- (volitelně) bonus: porovnání CBOW a skip-gram.

Natrénované vektory ani GloVe do repozitáře nenahrávejte, jsou příliš velké.
