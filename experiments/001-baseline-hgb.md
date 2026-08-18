# 001 — Baseline: HistGradientBoosting com 29 features

- **Data:** 2026-08-17
- **Commit:** `6ffa9a1` (código depois movido para `src/` sem mudança de resultado)
- **Baseia-se em:** do zero
- **CV:** 0,8110 ± 0,0092 — 5-fold estratificado, seed 42
- **LB público:** 0,80266 (submissão 55590027)

## Hipótese

Estabelecer um piso confiável antes de otimizar. A escolha do `HistGradientBoostingClassifier` não foi
por acaso: ele trata NaN e categóricas nativamente, e como ~2% de **toda** coluna está ausente nos dois
splits, evitar uma etapa de imputação arbitrária significa deixar o modelo aprender a direção do desvio
de cada ausência em vez de receber uma mediana inventada.

## O que mudou

Primeiro experimento. Features (29 no total):

- **Grupo:** `Group` e `GroupPos` extraídos de `PassengerId` (`gggg_pp`), `GroupSize`, `Solo`.
- **Cabine:** `Cabin` dividida em `Deck` / `CabinNum` / `Side`, mais `CabinSize` (quantas pessoas na
  mesma cabine exata).
- **Família:** `Surname` extraído de `Name`, com `FamilySize`. Pega famílias que o id de grupo separa.
- **Gastos:** `TotalSpend`, `SpendCount`, `NoSpend`, e `log1p` das cinco colunas de amenidade mais do
  total — elas são fortemente concentradas em zero e com cauda longa.
- **Ausências:** `NaNCount` por linha, testando o próprio padrão de ausência como sinal.
- **Idade:** `IsChild` (< 13 anos).

**A regra de domínio que mais rendeu:** passageiros em criosono ficam confinados à cabine e não
conseguem consumir nada. Isso torna duas imputações corretas, não apenas convenientes:

1. gasto ausente vira **zero** para quem está em criosono;
2. quem gastou qualquer coisa **não** estava em criosono.

São ~200 valores recuperados sem chute.

Hiperparâmetros: `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `min_samples_leaf=30`,
`l2_regularization=1.0`, early stopping com 10% de validação.

## Resultado

```
folds: [0.8125  0.7953  0.8200  0.8199  0.8072]
CV:    0.8110 +/- 0.0092
LB:    0.80266
```

Distribuição das predições: 51,7% `True`, próximo do balanço real do treino (50,4%) — o modelo não está
enviesado para uma classe.

**CV 0,8110 vs. LB 0,80266** — lacuna de ~0,008. Está dentro do desvio-padrão entre folds (0,0092), e o
leaderboard público cobre só parte do test set. Não há indício de overfitting relevante.

## Conclusão

Manter como baseline. O número já é competitivo: a maior parte do leaderboard fica em 0,80–0,81 e o topo
perto de 0,82, então a margem restante é estreita e os ganhos daqui em diante serão pequenos.

Nada foi descartado ainda — não há experimento anterior para comparar.

## Próximo passo

Em ordem de retorno esperado:

1. **Agregados de grupo mais ricos** — gasto total do grupo, se o grupo inteiro está em criosono,
   planeta dominante do grupo. Precisam ser calculados dentro de cada split, já que nenhum `gggg` cruza
   treino e teste; não há rótulo de companheiro para vazar.
2. **Busca de hiperparâmetros** com `RandomizedSearchCV` sobre `learning_rate`, `max_leaf_nodes` e
   `min_samples_leaf`.
3. **Ensemble** com LightGBM ou CatBoost, por média de probabilidades.

Antes de qualquer uma: confirmar a melhoria na CV. Com 10 submissões por dia, não vale gastar
tentativa em diferença menor que o ruído.
