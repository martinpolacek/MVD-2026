# Cvičení 1 - MetaCentrum a Git

Cílem cvičení je vyzkoušet si práci s výpočetním clusterem MetaCentrum a připravit si Git repozitář, do kterého budete během semestru odevzdávat všechna cvičení.

## Úloha 1 - Dávková úloha na MetaCentru

Napište Python skript, který pomocí knihovny NumPy vygeneruje matici náhodných čísel o rozměru 50 000 × 50 000 (typ `float64`, tj. cca 20 GB) a vypíše její průměr a směrodatnou odchylku. Výpočet je jednoduchý, ale kvůli velikosti matice ho na běžném počítači nespustíte.

Základy práce s NumPy:

```python
import numpy as np

rng = np.random.default_rng()
A = rng.random((1000, 1000))   # matice náhodných čísel z intervalu [0, 1)
v = rng.random(1000)           # vektor náhodných čísel

print(A.shape, A.dtype)        # rozměr a datový typ
print(A.nbytes / 1e9, "GB")    # velikost v paměti

print(A.mean(), A.std())       # průměr a směrodatná odchylka
print(A.sum(axis=0))           # součty sloupců

B = A * 2                      # násobení po prvcích
C = A @ A                      # maticové násobení
w = A @ v                      # násobení matice vektorem
```

Pozor, `A.std()` si interně vytvoří dočasnou kopii matice, takže potřebuje zhruba dvojnásobek paměti.

Skript spusťte na MetaCentru jako dávkovou úlohu. K tomu si připravte PBS skript, který si vyžádá dostatek paměti a skript spustí. Úlohu odešlete příkazem `qsub`.

## Úloha 2 - Git repozitář

Založte si na GitHubu repozitář s názvem `MVD-2026-Prijmeni` (např. `MVD-2026-Novak`) a nasdílejte ho s [@martinpolacek](https://github.com/martinpolacek). Do tohoto repozitáře budete odevzdávat i všechna další cvičení.

## Odevzdání

Do repozitáře nahrajte složku `cv01`, která bude obsahovat:
- Python skript,
- PBS skript,
- výstupní logy úlohy z MetaCentra.
