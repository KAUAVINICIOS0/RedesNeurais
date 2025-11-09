# RODES - Classificação Binária de Imagens (Urbano vs Natural)

## Descrição

A aplicação utiliza o dataset CIFAR-10 e realiza um remapeamento das classes originais:
- **Urbano (0)**: Automóveis, Caminhões, Navios
- **Natural (1)**: Aves, Gatos, Cervos, Cães, Cavalos, Sapos

O modelo é uma CNN simples que processa imagens de 32x32 pixels e produz métricas de avaliação, incluindo matriz de confusão, curva ROC e visualizações das previsões.

## Requisitos

### Python
- Python 3.8 ou superior

### Bibliotecas Necessárias
- `numpy` - Operações numéricas e arrays
- `matplotlib` - Visualização de gráficos e imagens
- `scikit-learn` - Métricas de avaliação e divisão de dados
- `seaborn` - Visualização estatística (heatmaps)
- `tensorflow` - Framework de deep learning
- `keras` - API de alto nível para TensorFlow

##  Instalação

### 1. Clone o repositório (se aplicável)
```bash
cd RODES
```

### 2. Crie um ambiente virtual (recomendado)
```bash
python3 -m venv venv
```

### 3. Ative o ambiente virtual

**Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### 4. Instale as dependências
```bash
pip install -r requirements.txt
```

Ou instale manualmente:
```bash
pip install numpy matplotlib scikit-learn seaborn tensorflow
```

## Como Executar

Após instalar todas as dependências, execute o script principal:

```bash
python main.py
```

## O que a aplicação faz

1. **Carrega o dataset CIFAR-10** - Baixa automaticamente na primeira execução
2. **Remapeia as classes** - Converte as 10 classes originais em 2 classes binárias
3. **Normaliza os dados** - Converte valores de pixel para o intervalo [0, 1]
4. **Divide os dados** - Separa em conjuntos de treino, validação e teste
5. **Treina o modelo** - CNN com 20 épocas
6. **Avalia o modelo** - Calcula acurácia, loss e outras métricas
7. **Gera visualizações**:
   - Matriz de confusão
   - Curva ROC com AUC
   - Gráficos de loss e acurácia durante o treinamento
   - Visualização de 12 imagens aleatórias com previsões

## Observações

- O dataset CIFAR-10 será baixado automaticamente na primeira execução (~170 MB)
- O treinamento pode levar alguns minutos dependendo do hardware
- As visualizações são exibidas em janelas separadas (matplotlib)
- Certifique-se de ter conexão com a internet na primeira execução para baixar o dataset

## Estrutura do Projeto

```
RODES/
├── main.py              # Script principal da aplicação
├── requirements.txt     # Dependências do projeto
├── README.md           # Este arquivo
└── venv/               # Ambiente virtual (não versionar)
```

## Solução de Problemas

### Erro ao importar TensorFlow/Keras
```bash
pip install --upgrade tensorflow
```

### Erro ao baixar o dataset
Certifique-se de ter conexão com a internet. O dataset será salvo em `~/.keras/datasets/` após o primeiro download.

### Problemas com visualizações (plt.show())
Se estiver em um ambiente sem interface gráfica, você pode salvar as figuras ao invés de exibi-las:
```python
plt.savefig('nome_do_arquivo.png')
```

