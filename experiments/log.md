# Notas de experimentos

Registro do que foi tentado, o que funcionou e o que não funcionou. Um arquivo por experimento,
numerado em sequência: `001-baseline-hgb.md`, `002-….md`. O modelo em branco está em
[`TEMPLATE.md`](TEMPLATE.md). Este arquivo é o índice.

A regra que faz essas notas valerem alguma coisa: **anote também o que falhou**. Saber que target
encoding não ajudou vale tanto quanto saber que o criosono ajudou — evita repetir o mesmo caminho
daqui a duas semanas.

## Resumo

| # | Experimento | CV (acurácia) | LB público | Δ vs. anterior | Commit | Submissão |
|---|---|---|---|---|---|---|
| [001](001-baseline-hgb.md) | HistGradientBoosting, 29 features | 0,8110 ± 0,0092 | **0,80266** | — | `6ffa9a1` | 55590027 |

## Referências de contexto

- CV local usa 5-fold estratificado, `random_state=42`. Números de CV só são comparáveis entre si se
  o esquema de validação for o mesmo — se mudar o número de folds ou a semente, diga isso na nota.
- O leaderboard público cobre apenas parte do test set. Com desvio-padrão de ~0,009 entre folds, uma
  diferença de 0,002–0,003 entre dois experimentos **não é sinal** — não persiga essas variações.
- Limite de 10 submissões por dia. Só submeta o que a CV indicar como melhoria real.
- Para gerar uma submissão nomeada por experimento: `python src/inference.py exp002.csv`.
