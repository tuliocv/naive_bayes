import re
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Laboratório Naive Bayes", page_icon="🧠", layout="wide")

SPAM = [
    ("Você ganhou um prêmio exclusivo, clique para resgatar agora", "Spam"),
    ("Parabéns você foi selecionado para receber dinheiro grátis", "Spam"),
    ("Oferta imperdível compre agora e ganhe um desconto especial", "Spam"),
    ("Seu prêmio em dinheiro está disponível confirme seus dados", "Spam"),
    ("Clique no link e receba seu bônus imediatamente", "Spam"),
    ("Última chance de participar do sorteio milionário", "Spam"),
    ("Você foi escolhido para ganhar um celular grátis", "Spam"),
    ("Resgate seu presente exclusivo antes que expire", "Spam"),
    ("Promoção relâmpago faça o pagamento e libere a recompensa", "Spam"),
    ("Crédito aprovado sem consulta solicite agora", "Spam"),
    ("Ganhe renda extra trabalhando poucas horas por dia", "Spam"),
    ("Seu bônus está bloqueado atualize sua conta pelo link", "Spam"),
    ("Investimento garantido com lucro imediato clique aqui", "Spam"),
    ("Urgente informe sua senha para desbloquear o prêmio", "Spam"),
    ("Você ganhou um vale compras resgate seu cupom", "Spam"),
    ("Sua conta será suspensa confirme seus dados neste endereço", "Spam"),
    ("Confira nosso sorteio ganhe prêmios incríveis hoje", "Spam"),
    ("Dinheiro fácil sem esforço cadastre-se já", "Spam"),
    ("Oferta limitada compre dois e receba dez gratuitamente", "Spam"),
    ("Acesse este link para liberar o depósito do seu prêmio", "Spam"),
    ("A reunião com a equipe será amanhã às dez horas", "Não spam"),
    ("Professor envio em anexo o relatório solicitado", "Não spam"),
    ("Lembrete sobre a entrega do trabalho na sexta feira", "Não spam"),
    ("Podemos reagendar nossa reunião para segunda feira", "Não spam"),
    ("Segue a ata da última reunião do departamento", "Não spam"),
    ("Seu pedido foi enviado e chegará na próxima semana", "Não spam"),
    ("Obrigado pela ajuda no projeto de pesquisa", "Não spam"),
    ("O material da aula já está disponível na plataforma", "Não spam"),
    ("Envio a versão revisada do documento em anexo", "Não spam"),
    ("Amanhã teremos apresentação dos seminários", "Não spam"),
    ("Confirme sua presença no encontro da equipe", "Não spam"),
    ("A biblioteca estará fechada no feriado", "Não spam"),
    ("O resultado da avaliação será divulgado na segunda", "Não spam"),
    ("Obrigado pelo retorno sobre o orçamento solicitado", "Não spam"),
    ("Reunião de alinhamento do curso às quatorze horas", "Não spam"),
    ("Você pode revisar o código antes da entrega", "Não spam"),
    ("O prêmio da olimpíada de matemática será entregue amanhã", "Não spam"),
    ("Compartilho o calendário acadêmico atualizado", "Não spam"),
    ("Gostaria de tirar uma dúvida sobre os exercícios", "Não spam"),
    ("O comprovante de matrícula está anexado ao e-mail", "Não spam"),
]

SENTIMENT = [
    ("Adorei o produto, excelente qualidade e ótimo atendimento", "Positivo"),
    ("O filme foi maravilhoso e muito emocionante", "Positivo"),
    ("Estou muito satisfeito com a compra", "Positivo"),
    ("Atendimento rápido, gentil e eficiente", "Positivo"),
    ("Experiência incrível, recomendo muito", "Positivo"),
    ("A comida estava deliciosa e o serviço foi ótimo", "Positivo"),
    ("O aplicativo é fácil de usar e funciona perfeitamente", "Positivo"),
    ("Entrega rápida e produto em perfeitas condições", "Positivo"),
    ("Gostei bastante da aula, explicação muito clara", "Positivo"),
    ("O suporte resolveu meu problema rapidamente", "Positivo"),
    ("Serviço excelente, superou minhas expectativas", "Positivo"),
    ("A equipe foi atenciosa e prestativa", "Positivo"),
    ("Muito bom, voltaria a comprar sem dúvida", "Positivo"),
    ("Tudo funcionou muito bem, estou feliz", "Positivo"),
    ("O curso foi interessante e muito útil", "Positivo"),
    ("Qualidade fantástica e preço justo", "Positivo"),
    ("Achei o atendimento maravilhoso", "Positivo"),
    ("O resultado foi melhor do que eu esperava", "Positivo"),
    ("Gostei da experiência e indicaria aos amigos", "Positivo"),
    ("Excelente organização e comunicação clara", "Positivo"),
    ("O atendimento foi horrível e muito demorado", "Negativo"),
    ("Detestei o produto, qualidade péssima", "Negativo"),
    ("O filme foi chato e decepcionante", "Negativo"),
    ("Estou insatisfeito com a compra", "Negativo"),
    ("Experiência terrível, não recomendo", "Negativo"),
    ("A comida estava fria e sem sabor", "Negativo"),
    ("O aplicativo trava o tempo todo", "Negativo"),
    ("Entrega atrasada e embalagem danificada", "Negativo"),
    ("A aula foi confusa e pouco esclarecedora", "Negativo"),
    ("O suporte ignorou meu problema", "Negativo"),
    ("Serviço ruim, não atendeu às expectativas", "Negativo"),
    ("A equipe foi grosseira e desorganizada", "Negativo"),
    ("Muito ruim, nunca mais voltaria", "Negativo"),
    ("Nada funcionou, fiquei frustrado", "Negativo"),
    ("O curso foi cansativo e inútil", "Negativo"),
    ("Preço abusivo e qualidade baixa", "Negativo"),
    ("Achei o atendimento péssimo", "Negativo"),
    ("O resultado foi pior do que eu esperava", "Negativo"),
    ("Não gostei da experiência e não indicaria", "Negativo"),
    ("Péssima organização e informações confusas", "Negativo"),
]

DATASETS = {"📨 Detector de spam": SPAM, "💬 Análise de sentimentos": SENTIMENT}

@st.cache_resource
def train(kind, representation, alpha):
    data = DATASETS[kind]
    Vectorizer = CountVectorizer if representation == "Bag of Words" else TfidfVectorizer
    pipe = Pipeline([
        ("vetorizar", Vectorizer(lowercase=True, strip_accents="unicode")),
        ("naive_bayes", MultinomialNB(alpha=alpha)),
    ])
    pipe.fit([t for t, _ in data], [label for _, label in data])
    return pipe

st.title("🧠 Laboratório de Inteligência Artificial")
st.write("**Naive Bayes aplicado à classificação de textos** — experimente e investigue os erros do modelo.")
st.info("Os dados são sintéticos e pequenos: o projeto é exclusivamente didático. Probabilidades estimadas pelo modelo não são garantias de acerto.")

with st.sidebar:
    st.header("⚙️ Configurações")
    representation = st.radio("Representação do texto", ["Bag of Words", "TF-IDF"])
    alpha = st.slider("Suavização de Laplace (α)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
    st.caption("Altere os parâmetros e veja como a classificação muda.")

tabs = st.tabs(list(DATASETS.keys()) + ["📚 Sobre os dados"])
for tab, kind in zip(tabs[:2], DATASETS):
    with tab:
        model = train(kind, representation, alpha)
        key = "spam" if "spam" in kind else "sentimento"
        default = "Você ganhou um prêmio, clique aqui para resgatar" if key == "spam" else "O atendimento foi excelente e a entrega muito rápida"
        message = st.text_area("Digite um texto para classificar:", default, height=110, key=f"input_{key}")
        if message.strip():
            label = model.predict([message])[0]
            probabilities = model.predict_proba([message])[0]
            classes = model.named_steps["naive_bayes"].classes_
            confidence = float(max(probabilities))
            col1, col2 = st.columns(2)
            col1.metric("Classificação prevista", label)
            col2.metric("Maior probabilidade estimada", f"{confidence:.1%}")
            dfp = pd.DataFrame({"Classe": classes, "Probabilidade": probabilities}).set_index("Classe")
            st.bar_chart(dfp)
            vectorizer = model.named_steps["vetorizar"]
            terms = set(vectorizer.get_feature_names_out())
            words = re.findall(r"\b\w+\b", message.lower())
            recognized = [w for w in words if w in terms]  # accents may be normalized
            vectorized = vectorizer.transform([message])
            recognized_terms = vectorizer.get_feature_names_out()[vectorized.nonzero()[1]].tolist()
            st.write("**Palavras presentes no vocabulário do modelo:**", ", ".join(recognized_terms) if recognized_terms else "Nenhuma")
            if vectorized.nnz == 0:
                st.warning("Nenhuma palavra foi reconhecida. A previsão depende apenas das frequências das classes no treinamento.")
            st.caption("A classificação depende das frases usadas para treinar este modelo, não de uma compreensão humana do texto.")
        else:
            st.warning("Digite alguma mensagem para obter uma classificação.")
        with st.expander("Ver exemplos de treinamento"):
            st.dataframe(pd.DataFrame(DATASETS[kind], columns=["Texto", "Classe"]), use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("De onde vem a inteligência deste sistema?")
    st.markdown("""
    1. **Dados rotulados:** exemplos de textos previamente classificados.
    2. **Vetorização:** transforma palavras em contagens (Bag of Words) ou pesos (TF-IDF).
    3. **Naive Bayes:** aprende frequências condicionais das palavras por classe.
    4. **Predição:** calcula probabilidades relativas e escolhe a classe mais provável.
    """)
    st.write("**Perguntas para o laboratório:**")
    st.markdown("- O que ocorre ao escrever 'não foi ruim'?\n- A frase 'ganhou prêmio' sempre indica spam?\n- O resultado muda ao trocar Bag of Words por TF-IDF?\n- O que acontece quando nenhuma palavra é conhecida?")
    st.write("**Dimensão dos conjuntos de treinamento:**")
    for name, rows in DATASETS.items():
        st.write(f"{name}: {len(rows)} exemplos")
