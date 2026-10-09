# Laboratório Naive Bayes: Spam e Sentimentos

## Instalação (Python 3.10+)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Abra no navegador o endereço local mostrado pelo Streamlit (em geral, http://localhost:8501).

## Como usar

- Escolha a aba de spam ou de análise de sentimentos.
- Digite uma frase e compare a classificação e probabilidades.
- Mude Bag of Words / TF-IDF e o parâmetro alfa na barra lateral.
- Veja os exemplos usados no treinamento e teste novas frases.

## Limitações

Dados de treinamento sintéticos e pequenos. O projeto serve para ensino, não é um serviço de moderação de conteúdo ou detecção de golpes real. O app treina com todos os dados fornecidos, sem separar conjunto de teste; as probabilidades exibidas não foram calibradas. Para avaliar o desempenho, faça uma separação treino/teste ou validação cruzada com dados adicionais e representativos.
