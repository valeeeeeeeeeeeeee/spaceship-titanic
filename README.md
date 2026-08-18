# Spaceship Titanic — Kaggle

Solução para a competição [Spaceship Titanic](https://www.kaggle.com/c/spaceship-titanic) (Getting
Started, patrocinada pela Google LLC).

**A tarefa:** prever se um passageiro foi transportado para uma dimensão alternativa durante a colisão
da Spaceship Titanic com uma anomalia do espaço-tempo. Classificação binária sobre `Transported`, a
partir de registros pessoais recuperados do sistema de bordo danificado.

> **Status:** apenas dados e documentação. Ainda não há código de modelagem.

## Dados

Os CSVs **não são versionados** — as regras da competição proíbem redistribuir os dados a quem não
aceitou os termos. Baixe-os direto da Kaggle:

```python
import kagglehub
path = kagglehub.competition_download('spaceship-titanic')
```

Depois copie `train.csv`, `test.csv` e `sample_submission.csv` do caminho retornado
(`~/.cache/kagglehub/competitions/spaceship-titanic/`) para a raiz deste repositório.

É preciso um token da API da Kaggle no ambiente e a conta precisa ter aceitado as regras na página da
competição, senão o download retorna 403:

```bash
# formato novo (KGAT_…), dispensa username
export KAGGLE_API_TOKEN="KGAT_…"
# ou o formato legado: ~/.kaggle/kaggle.json
```

O token nunca deve ser gravado em nenhum arquivo do repositório.

### Estrutura dos arquivos

| Arquivo | Linhas | Colunas | Descrição |
|---|---|---|---|
| `train.csv` | 8693 | 14 | ~2/3 dos passageiros, com o alvo `Transported` (~50/50, 50,4% True) |
| `test.csv` | 4277 | 13 | ~1/3 dos passageiros, idêntico ao treino menos o alvo |
| `sample_submission.csv` | 4277 | 2 | formato de submissão, já na mesma ordem de linhas do `test.csv` |

### Colunas

- **`PassengerId`** — id único no formato `gggg_pp`, onde `gggg` é o grupo com que o passageiro viaja e
  `pp` é seu número dentro do grupo. Membros de um grupo costumam ser familiares, mas nem sempre.
- **`HomePlanet`** — planeta de origem, em geral o de residência permanente. *Europa, Earth, Mars.*
- **`CryoSleep`** — se o passageiro optou por animação suspensa durante a viagem. Quem está em criosono
  fica **confinado à cabine** e por isso não consome nenhuma amenidade.
- **`Cabin`** — `deck/num/side`, onde *side* é `P` (Port) ou `S` (Starboard). Decks de `A` a `G`, mais `T`.
- **`Destination`** — planeta de desembarque. *TRAPPIST-1e, PSO J318.5-22, 55 Cancri e.*
- **`Age`** — idade do passageiro.
- **`VIP`** — se pagou pelo serviço VIP da viagem.
- **`RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`** — valor gasto em cada amenidade de luxo
  da nave. Fortemente concentrados em zero.
- **`Name`** — nome e sobrenome. Presente também no teste, então agregados por sobrenome são viáveis.
- **`Transported`** — se o passageiro foi transportado para outra dimensão. **É o alvo.**

### Dois detalhes medidos nos dados

- **A divisão treino/teste é por grupo:** nenhum `gggg` aparece nos dois arquivos. Não dá para usar o
  rótulo conhecido de um companheiro de grupo — features de grupo precisam ser agregados internos
  (tamanho, gasto, planeta), sem vazamento do alvo.
- **Valores ausentes em toda parte:** cada coluna exceto `PassengerId` tem ~2% de NaN nos dois splits
  (179–217 por coluna no treino, 80–106 no teste). Imputação é obrigatória, e vale testar o próprio
  padrão de ausência como feature.

## Ambiente

Python 3.14 com:

```
pandas 2.3.3   scikit-learn 1.9.0   numpy 2.3.5   kagglehub 1.0.2
```

## Submissão

O arquivo precisa ter duas colunas — `PassengerId` e `Transported` — com 4277 linhas na mesma ordem do
`test.csv`. `Transported` deve ser serializado como `True`/`False`, não `1`/`0`.

O limite é de **10 submissões por dia**.

## Regras da competição

O texto oficial completo está em [`COMPETITION_RULES.md`](COMPETITION_RULES.md). Os pontos que afetam o
código deste repositório:

- Não é permitido **rotulagem manual** nem predição humana dos registros de teste.
- Código da competição **não pode ser compartilhado privadamente** fora da equipe; o compartilhamento
  público deve ocorrer no fórum ou nos notebooks da Kaggle, sob licença aprovada pela OSI.
- Dependências open source precisam de licença aprovada pela OSI, sem restrição de uso comercial.
- **Dados externos** só são permitidos se forem públicos e igualmente acessíveis a todos os
  participantes, sem custo.
- Ferramentas de **AutoML** são explicitamente permitidas.
- Os dados não podem ser redistribuídos — daí o `.gitignore`.

## Estrutura do repositório

```
├── CLAUDE.md              orientações para o Claude Code
├── COMPETITION_RULES.md   regras oficiais na íntegra
├── README.md
├── .gitignore             ignora dados, submissões, credenciais e modelos
└── (CSVs baixados da Kaggle, não versionados)
```
