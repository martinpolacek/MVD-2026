# Cvičení 7 - K-means a samoorganizující se mapa

Úloha 1: K-means na umělých datech, lokálně. Úloha 2: samoorganizující se neuronová síť (SOM) učená po vzorku nad vektory Wikipedie na MetaCentru.

Potřebné knihovny: `numpy`, `scikit-learn`, `matplotlib`, `torch`, `datasets`

## Úloha 1 - K-means

Vytvořte čtyři datasety a zobrazte je:

```python
from sklearn.datasets import make_blobs, make_moons, make_circles

D1 = make_blobs(n_samples=100, centers=2, cluster_std=1.0, random_state=7)
D2 = make_blobs(n_samples=200, centers=4, cluster_std=1.0, random_state=7)
D3 = make_moons(n_samples=200, noise=0.05, random_state=7)
D4 = make_circles(n_samples=200, noise=0.03, factor=0.5, random_state=7)   # každý je dvojice (X, y)
```

Skutečné třídy `y` algoritmu nedávejte. Slouží jen k vyhodnocení pomocí `adjusted_rand_score(y, labels)` ze `sklearn.metrics` (ARI: 1 = shoda s třídami, kolem 0 = jako náhodné rozdělení). Počet shluků K nastavte podle počtu tříd.

1. Naprogramujte **K-means** v `numpy`, vektorizovaně (žádné cykly přes body). Centroidy zvolte jako K náhodných bodů. Střídejte přiřazení k nejbližšímu centroidu a výpočet nových centroidů, dokud kritérium J neklesne o méně než 10<sup>−6</sup>:

```math
J = \sum_{j=1}^{K} \sum_{i=1}^{N_j} \left\| x_i^{(j)} - \mu_j \right\|^2
```

2. Funkce vrátí přiřazení, centroidy a hodnotu J v každé iteraci. Ověřte, že J nikdy neroste. `KMeans` ze `sklearn` použijte jen pro kontrolu.
3. Každý dataset spusťte 10krát z různých startů a nechte výsledek s nejmenším J. Zobrazte ho a centroidy zvýrazněte.
4. U datasetu 2 vykreslete J podle iterace a vypište nejmenší a největší J z 10 startů.

Kontrola:

| Dataset | J | ARI |
|---|---|---|
| 1 | 194,63 | 1,00 |
| 2 | 384,85 | 1,00 |
| 3 | 80,39 | 0,24 |
| 4 | asi 78,9 | asi 0 |

Ukázka výsledku: [`sample_output.png`](sample_output.png)

## Úloha 2 - Samoorganizující se neuronová síť

### Jak síť funguje

Síť (SOM) tvoří G × G neuronů. Síť 40 × 40 tedy znamená 1600 neuronů, ne rozměr vstupu: pro vektory o 384 rozměrech jsou váhy matice 1600 × 384. Každý neuron má dvě různé polohy:
- **vektor vah** w<sub>i</sub> o stejném rozměru jako data, tedy polohu v prostoru dat. Ta se učením mění,
- **místo v mřížce** (řádek, sloupec). To je pevné a určuje, kdo je čí soused.

Trénovací data se zpracovávají **vzorek po vzorku**. Pro jeden vzorek x<sub>j</sub>:

1. Spočítejte vzdálenost ‖w<sub>i</sub> − x<sub>j</sub>‖ ke všem neuronům. Nejbližší neuron je vítěz (best).
2. Podle mřížky určete region R: neurony, pro které |Δřádek| ≤ r a |Δsloupec| ≤ r, kromě vítěze. Poloměr r se měří v mřížce, ne v prostoru dat.
3. Posuňte váhy vítěze i regionu směrem ke vzorku:

```math
w_{t+1,best} = w_{t,best} + \alpha\,(x_j - w_{t,best}), \qquad w_{t+1,r} = w_{t,r} + \beta\,(x_j - w_{t,r})
```

**α** říká, jak velkou část vzdálenosti ke vzorku vítěz urazí: α = 0 znamená žádný pohyb, α = 1 skok přímo na vzorek. **β** je totéž pro neurony v regionu. Sousedé se mají posunout méně, proto β < α. Tím, že se sousedé pohybují společně s vítězem, si sousední neurony v mřížce zůstanou podobné a síť se sama uspořádá.

**Nastavení.** Na začátku se síť hýbe hodně a celá, aby se uspořádala. Na konci se každý neuron jen jemně doladí. Označte t = počet zpracovaných vzorků, T = počet vzorků × počet epoch (epocha je jeden průchod všemi daty):

```math
\alpha = 0{,}5 + (0{,}02 - 0{,}5)\,\frac{t}{T}, \qquad \beta = \frac{\alpha}{2}, \qquad r = \frac{G}{2}\left(1 - \frac{t}{T}\right)
```

| Průběh učení | α | β | r (G = 40) |
|---|---|---|---|
| začátek | 0,50 | 0,25 | 20 |
| polovina | 0,26 | 0,13 | 10 |
| konec | 0,02 | 0,01 | 0 (posouvá se jen vítěz) |

Na začátku je každý neuron kopií vektoru jednoho náhodně vybraného článku. Pro síť 40 × 40 tedy vyberete 1600 různých článků a jejich vektory budou počáteční váhy: `W = X[torch.randperm(len(X), device=device)[:G * G]].clone()`, kde `X` je tenzor s vektory. V každé epoše vzorky zamíchejte.

**Programování.** Cyklus přes vzorky je tady správně, to je podstata učení po vzorku. Výpočet pro jeden vzorek ale vektorizujte (žádný cyklus přes neurony). Hotovou knihovnu pro SOM nepoužívejte, učení napište v PyTorchi, aby běželo na GPU. Nebude v něm žádný `backward` ani optimizer: síť nemá ztrátovou funkci ani gradient, váhy měníte přímo podle vzorce. Váhy jsou obyčejný tenzor bez `requires_grad`.

**Mřížka.** Neurony jsou očíslované po řádcích: neuron číslo `k` leží na řádku `k // G`, sloupci `k % G`. Region pro aktuální vzorek určíte takto:

```python
radek = torch.arange(G * G, device=device) // G       # místo neuronů v mřížce, stačí spočítat jednou před učením
sloupec = torch.arange(G * G, device=device) % G

region = ((radek - radek[best]).abs() <= r) & ((sloupec - sloupec[best]).abs() <= r)   # znovu pro každý vzorek
region[best] = False                                  # vítěz se posouvá zvlášť, o α
```

### Mapa Wikipedie

Použijte vektory Simple Wikipedie z 5. cvičení. Titulky mají stejné pořadí jako vektory:

```python
import numpy as np
from datasets import load_dataset

X = np.load("wiki_emb.npy").astype(np.float32)     # 241 787 x 384
wiki = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train")
titles, texty = wiki["title"], wiki["text"]
```

Kód si nejdřív vyzkoušejte na malé části dat (např. 5000 náhodných článků a síť 10 × 10).

1. Napište skript, který naučí síť **40 × 40** (3 epochy) a uloží váhy do `som_W.npy`. Spusťte ho na MetaCentru jako dávkovou úlohu s GPU (fronta `gpu`, `ngpus=1`).

2. Každý článek přiřaďte k nejbližšímu neuronu:

```python
W = np.load("som_W.npy")                           # 1600 x 384

d2 = (X ** 2).sum(1, keepdims=True) - 2 * X @ W.T + (W ** 2).sum(1)   # ‖x − w‖² = ‖x‖² − 2 x·w + ‖w‖²
clanek_neuron = d2.argmin(1)
vzdalenost = np.sqrt(np.maximum(d2.min(1), 0))

pocty = np.bincount(clanek_neuron, minlength=len(W))
```

   Vypište průměrnou vzdálenost a nejmenší, mediánový a největší počet článků na neuron.

3. Váhy neuronů rozdělte vaším K-means z úlohy 1 do K = 8 oblastí. Stejně jako tam ho spusťte 10krát a nechte výsledek s nejmenším J. Váhy jsou tu data: 1600 bodů o 384 rozměrech. Výsledek uložte do `oblast_neuronu`. Pro každou oblast vypište typická slova, tedy slova častější v oblasti než jinde:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer(stop_words="english", min_df=20, max_df=0.2, sublinear_tf=True)
M = tf.fit_transform([f"{t}. {x[:600]}" for t, x in zip(titles, texty)])
slova = tf.get_feature_names_out()
vse = M.mean(axis=0).A1

oblast_clanku = oblast_neuronu[clanek_neuron]
for k in range(8):
    skore = M[oblast_clanku == k].mean(axis=0).A1 - vse
    print(k, slova[skore.argsort()[::-1][:6]])
```

4. Podle typických slov oblasti pojmenujte (např. `football, league, team` → „Sport“). Pak nakreslete **mapu**: jedno políčko je jeden neuron, barva je oblast a bílé čáry ukazují mřížku sítě. Vektory o 384 číslech se do obrázku nekreslí: poloha políčka je místo neuronu v mřížce, z vektoru se bere jen barva (číslo oblasti). Mapu nejdřív vykreslete bez názvů. Podle obrázku pak přidejte pro každou oblast jeden řádek `ax.text` se sloupcem a řádkem, kde daná barva leží. Uložte jako `som_wiki_mapa.png`:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 9))
ax.imshow(oblast_neuronu.reshape(40, 40), cmap="tab10", vmin=0, vmax=9)
ax.set_xticks(np.arange(-0.5, 40), minor=True)
ax.set_yticks(np.arange(-0.5, 40), minor=True)
ax.grid(which="minor", color="white", linewidth=1)
ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
ax.text(12, 5, "Sport", ha="center", va="center", backgroundcolor="white")   # sloupec, řádek, název; souřadnice jsou jen příklad
# ... stejný řádek pro každou další oblast
fig.savefig("som_wiki_mapa.png", dpi=110)
```

   Ukázka: [`sample_wiki_mapa.png`](sample_wiki_mapa.png). Vaše oblasti a názvy se budou lišit a legenda vpravo není povinná.

Kontrola kroků 1 a 2:
- průměrná vzdálenost asi 0,80,
- žádný neuron bez článku, medián asi 140 článků na neuron, největší asi 550 až 570.

## Odevzdání

Složka `cv07` ve vašem repozitáři:
- notebook s úlohou 1 a s kroky 2 až 4 úlohy 2,
- obrázek `som_wiki_mapa.png`,
- skript a PBS skript k učení sítě a log z MetaCentra.

Vektory ani váhy sítě do repozitáře nenahrávejte.
