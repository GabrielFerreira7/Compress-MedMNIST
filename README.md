# 🔧 Neural Network Compression with NSGA-III

Este projeto realiza compressão de redes neurais convolucionais por meio de **poda** e **quantização**, utilizando algoritmos de **otimização multiobjetivo (MO)** para buscar o melhor trade-off entre **acurácia** e **eficiência computacional (FLOPs)**. A abordagem emprega o algoritmo **NSGA-III** da biblioteca `pymoo`.

---

## 📁 Estrutura do Projeto

.
├── nsga.py # Script principal de execução  
├── compression.py # Funções auxiliares de compressão e avaliação  
├── prunne.py # Funções específicas de poda de modelos  
├── problem.py # Definição do problema para a otimização  
└── MOS2/ # Diretório de saída com resultados (criado dinamicamente)

---

## 🚀 Como Executar

### ✅ Requisitos

- Python 3.8+  
- pymoo  
- torch  
- numpy  
- scikit-learn  
- torchvision  

Instale as dependências com:

```bash
pip install pymoo torch torchvision numpy scikit-learn
```

### ▶️ Execução

O script principal é `nsga.py` e deve ser executado via terminal com um argumento indicando o conjunto de dados e arquitetura:

```bash
python nsga.py <ID>
```

#### IDs Suportados

| ID | Dataset   | Arquitetura     |
|----|-----------|-----------------|
| 1  | CIFAR-10  | ResNet50        |
| 2  | Blood     | ResNet50        |
| 3  | Derma     | ResNet50        |
| 4  | Retina    | ResNet50        |
| 5  | CIFAR-10  | MobileNet_V2    |
| 6  | Blood     | MobileNet_V2    |
| 7  | Derma     | MobileNet_V2    |
| 8  | Retina    | MobileNet_V2    |

---

## ⚙️ Lógica do Projeto

### nsga.py
- Define o número de camadas com base no argumento de entrada.  
- Inicializa o problema de compressão (`problem_compress`).  
- Executa o algoritmo NSGA-III com critérios definidos.  
- Salva os resultados (indivíduos, objetivos e tempo) em arquivos `.pkl`.  
- Aplica a compressão com poda e quantização usando os indivíduos finais.  

### problem.py
- Implementa `problem_compress`, uma subclasse de `ElementwiseProblem` do `pymoo`.  
- Define a função objetivo como uma tupla `(acc, flops)` com base na compressão realizada.  

### compression.py
- Contém funções para avaliação de acurácia, aplicação de compressão, e organização dos resultados.  
- Chama `eval_compression` (do `prunne.py`) para aplicar as máscaras de poda nas camadas.  

### prunne.py
- Define funções para aplicação real de poda estrutural (L1) nas camadas convolucionais e lineares.  
- Calcula FLOPs ajustados após compressão.  

---

## 📦 Saída

Os resultados são salvos no diretório `MOS2/`, contendo:

- `Resultados/sn/*Res.pkl`: objeto de resultado completo do `pymoo`.  
- `Variaveis/Poda/sn/*.pkl`: resultados da compressão com poda.  
- `Variaveis/Quantizacao/sn/*.pkl`: resultados da compressão com quantização.  

---

## 📈 Objetivos de Otimização

- **Objetivo 1 (F1):** Minimizar a perda de acurácia após compressão.  
- **Objetivo 2 (F2):** Minimizar a complexidade computacional (FLOPs) do modelo.  

---

## 🧠 Extensões Futuras

- Adicionar novas arquiteturas como EfficientNet ou DenseNet.  
- Suporte a quantização-aware training (QAT).  
- Visualização dos Frentes de Pareto com `matplotlib` ou `pymoo`.  

---

## 👨‍💻 Autor

**Gabriel Ferreira**  
Mestrando em Ciência da Computação — UFOP  
Especialista em compressão de redes neurais e otimização multiobjetivo
