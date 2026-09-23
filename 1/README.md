# Cvičení 1 - MetaCentrum a Git

**Cíl:** Naučit se spustit dávkovou úlohu na MetaCentru a odevzdávat cvičení přes Git.

## Úloha 1 - Dávková úloha na MetaCentru

Napište krátký Python skript, který spočítá něco jednoduchého, ale potřebuje **alespoň 20 GB RAM**. Takovou úlohu na běžném notebooku nespustíte, a proto ji spustíte jako dávkovou úlohu na MetaCentru.

**Postup:**
1. Přihlaste se na frontend MetaCentra (např. `ssh uzJmeno@skirit.metacentrum.cz`).
2. Připravte si Python prostředí s knihovnou `numpy`, například pomocí skriptu [metacentrum_install.sh](metacentrum_install.sh).
3. Napište Python skript `uloha.py`, který:
   - vytvoří v paměti velké pole náhodných čísel o velikosti alespoň 20 GB (nápověda: matice `50 000 × 50 000` typu `float64` má 20 GB),
   - spočítá nad ním něco rychlého (např. průměr, směrodatnou odchylku, součet řádků),
   - vypíše výsledky a špičkové využití paměti (např. `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss`).
4. Napište PBS skript `uloha.sh`, který:
   - si vyžádá dostatek paměti (s rezervou, např. `mem=24gb`), 1 CPU a krátký `walltime` (např. 30 minut),
   - aktivuje Python prostředí a spustí `uloha.py`.
5. Odešlete úlohu pomocí `qsub uloha.sh` a sledujte její stav pomocí `qstat -u $USER`.
6. Po doběhnutí uložte výpis využitých prostředků: `qstat -xf <JOBID> > qstat.txt`.

Ukázky PBS skriptů a přehled práce s MetaCentrem najdete v přednášce 1.

## Úloha 2 - Git repozitář pro odevzdávání

1. Založte si na GitHubu repozitář s názvem **`MVD-2026-Prijmeni`** (např. `MVD-2026-Novak`).
2. Nasdílejte ho s uživatelem **[@martinpolacek](https://github.com/martinpolacek)** (*Settings → Collaborators → Add people*). To je nutné hlavně u soukromého repozitáře.
3. Do repozitáře nahrajte složku `cv01` s tímto obsahem:
   - `uloha.py` - Python skript,
   - `uloha.sh` - PBS skript,
   - výstupy úlohy (soubory `*.o<JOBID>` a `*.e<JOBID>`),
   - `qstat.txt` - výpis `qstat -xf`, ze kterého je vidět využitá paměť (`resources_used.mem`).

Do tohoto repozitáře budete odevzdávat i všechna další cvičení, každé do vlastní složky (`cv02`, `cv03`, ...).
