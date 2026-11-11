# Cvičení 8 - DBSCAN a spektrální shlukování

Úloha 1: DBSCAN. Úloha 2: spektrální shlukování. Úloha 3: porovnání s K-means a čistota. Vše lokálně na stejných umělých datech jako v 7. cvičení.

Potřebné knihovny: `numpy`, `scikit-learn`, `matplotlib`

## Data

```python
from sklearn.datasets import make_blobs, make_moons, make_circles

D1 = make_blobs(n_samples=100, centers=2, cluster_std=1.0, random_state=7)
D2 = make_blobs(n_samples=200, centers=4, cluster_std=1.0, random_state=7)
D3 = make_moons(n_samples=200, noise=0.05, random_state=7)
D4 = make_circles(n_samples=200, noise=0.03, factor=0.5, random_state=7)   # každý je dvojice (X, y)
```

Skutečné třídy `y` algoritmům nedávejte. Slouží jen k vyhodnocení pomocí `adjusted_rand_score(y, labels)` ze `sklearn.metrics`.

Hotové shlukovací algoritmy ze `sklearn` používejte jen pro kontrolu. Vzdálenosti všech dvojic bodů počítejte vektorizovaně (matice N × N, žádný cyklus přes dvojice bodů).

## Úloha 1 - DBSCAN

1. Naprogramujte funkci `dbscan(X, eps, min_pts)`, která vrátí čísla shluků bodů (0, 1, …; šum = −1) a masku core bodů. Postup podle přednášky:
   - **Okolí Eps** bodu jsou body ve vzdálenosti nejvýš Eps, včetně bodu samotného.
   - **Core bod** má v okolí Eps aspoň MinPts bodů.
   - Procházejte body. Když bod ještě nemá shluk a je core, založte nový shluk a rozšiřujte ho: dejte bod do fronty, z fronty postupně vybírejte body a sousedy každého **core** bodu bez shluku přidejte do shluku a do fronty.
   - **Border bod** (není core, ale leží v okolí core bodu) do shluku patří, ale shluk dál nerozšiřuje.
   - Body, které zůstanou bez shluku, jsou **šum** (−1).

   Cyklus přes body a fronta jsou tady v pořádku. Matici sousedů ale spočítejte jednou předem a vektorizovaně.

2. Spusťte DBSCAN s **MinPts = 5** a s Eps podle tabulky. Výsledky zobrazte, šum zvlášť (např. šedé křížky). Vypište počet shluků, počet šumu a ARI.

3. Porovnejte se `sklearn.cluster.DBSCAN(eps=…, min_samples=5)`: core body (`core_sample_indices_`) i šum musí být stejné. Čísla shluků se mohou lišit pořadím, proto shodu ověřte přes `adjusted_rand_score(labels_sklearn, labels)`, které musí vyjít 1.

4. U datasetu 4 zkuste ještě Eps = 0,15 a Eps = 0,3 a vypište totéž.

Kontrola:

| Dataset | Eps | Shluků | Šum | ARI |
|---|---|---|---|---|
| 1 | 1,0 | 2 | 2 | 0,96 |
| 2 | 1,0 | 4 | 7 | 0,95 |
| 3 | 0,2 | 2 | 0 | 1,00 |
| 4 | 0,2 | 2 | 0 | 1,00 |
| 4 | 0,15 | 12 | 6 | 0,54 |
| 4 | 0,3 | 2 | 0 | 1,00 |

## Úloha 2 - Spektrální shlukování

1. Naprogramujte funkci `spektralni(X, K, sigma)` podle kroků z přednášky. Začíná maticí podobnosti, která má na diagonále nuly:

```math
W_{ij} = \exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\right), \qquad W_{ii} = 0
```

   Další kroky:
   - matice stupňů D je diagonální, D<sub>ii</sub> = součet i-tého řádku W,
   - Laplacián L = D − W,
   - vlastní čísla a vektory: `vals, vecs = np.linalg.eigh(L)`. Vlastní čísla jsou seřazená od nejmenšího a vlastní vektory jsou **sloupce** `vecs`. Prvních K sloupců tvoří matici V<sub>k</sub>, řádek i je nová souřadnice bodu i,
   - na řádcích V<sub>k</sub> spusťte váš K-means ze 7. cvičení (10 startů, nechte výsledek s nejmenším J).

   Funkce vrátí čísla shluků, vlastní čísla a V<sub>k</sub>.

2. Spusťte ji na všech čtyřech datasetech s K podle počtu tříd a **σ = 1,0** (datasety 1 a 2), **σ = 0,1** (datasety 3 a 4). Vypište K + 1 nejmenších vlastních čísel a ARI.

3. U datasetů 3 a 4 vykreslete řádky V<sub>k</sub>: 1. sloupec na ose x, 2. sloupec na ose y. Rozsah obou os nastavte pevně na −0,15 až 0,15, jinak matplotlib roztáhne nepatrné rozdíly přes celý graf. Body každé skupiny by měly splynout do jednoho místa. Ukázka: [`sample_spektral.png`](sample_spektral.png), poloha a znaménka se u vás mohou lišit.

4. U datasetu 4 zkuste ještě σ = 0,2 a vypište totéž.

Kontrola:

| Dataset | σ | K + 1 nejmenších vlastních čísel | ARI |
|---|---|---|---|
| 1 | 1,0 | 0; 0,0022; 3,60 | 1,00 |
| 2 | 1,0 | 0; 0; 0; 0,0022; 0,89 | 1,00 |
| 3 | 0,1 | 0; 0; 0,025 | 1,00 |
| 4 | 0,1 | 0; 0,0004; 0,016 | 1,00 |
| 4 | 0,2 | 0; 0,50; 0,52 | asi 0 |

Čísla blízko nule se mohou vypsat jako malá záporná (např. −1e−15), to je zaokrouhlovací chyba.

## Úloha 3 - Porovnání a čistota

1. Napište funkci `cistota(y, labels)`: pro každý shluk vezměte počet bodů převládající třídy, sečtěte a vydělte počtem všech bodů. Šum (−1) do žádného shluku nepatří: v čitateli se nepočítá, ve jmenovateli ano.

```math
\text{čistota} = \frac{1}{n} \sum_{i} \max_j n_{ij}
```

   kde n<sub>ij</sub> je počet bodů třídy j ve shluku i.

2. Pro všechny čtyři datasety porovnejte K-means ze 7. cvičení, DBSCAN z úlohy 1 a spektrální shlukování z úlohy 2 (parametry jako výše). Vypište tabulku ARI a čistoty a výsledky zobrazte v mřížce 3 × 4 (řádky metody, sloupce datasety). Ukázka: [`sample_output.png`](sample_output.png).

Kontrola (ARI / čistota):

| Dataset | K-means | DBSCAN | Spektrální |
|---|---|---|---|
| 1 | 1,00 / 1,00 | 0,96 / 0,98 | 1,00 / 1,00 |
| 2 | 1,00 / 1,00 | 0,95 / 0,96 | 1,00 / 1,00 |
| 3 | 0,24 / 0,74 | 1,00 / 1,00 | 1,00 / 1,00 |
| 4 | asi 0 / asi 0,5 | 1,00 / 1,00 | 1,00 / 1,00 |

## Odevzdání

Složka `cv08` ve vašem repozitáři: notebook se všemi třemi úlohami a s vykreslenými obrázky.
