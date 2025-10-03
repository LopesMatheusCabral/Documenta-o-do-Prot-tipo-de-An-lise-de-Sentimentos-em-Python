from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from joblib import dump, load
import os
import json

# Arquivos para salvar o modelo, vetorizador e dados
model_file = 'sentiment_model.joblib'
vectorizer_file = 'vectorizer.joblib'
data_file = 'sentiment_data.json'

# Dados de treinamento
default_frases = [
    "O atendimento foi horrível, nunca mais volto.",
    "Excelente experiência, recomendo a todos.",
    "Demorou demais para chegar.",
    "O produto é bom e chegou rápido.",
    "Funcionários atenciosos e prestativos.",
    "Péssima qualidade, não recomendo.",
    "Muito feliz com a compra, parabéns à equipe!",
    "Estou extremamente insatisfeito com a compra.",
    "A entrega foi rápida e eficiente.",
    "Não funcionou como prometido.",
    "O suporte técnico resolveu meu problema rapidamente.",
    "O site é confuso e difícil de usar.",
    "Superou minhas expectativas!",
    "Fiquei decepcionado com o resultado.",
    "Preço justo e qualidade excelente.",
    "Funcionários mal-educados e despreparados.",
    "O sistema trava toda hora, péssimo.",
    "Adorei, tudo funcionou perfeitamente!",
    "Experiência frustrante, não recomendo.",
    "Voltarei a comprar com certeza.",
]

default_labels = [
    'NEGATIVA', 'POSITIVA', 'NEGATIVA', 'POSITIVA', 'POSITIVA',
    'NEGATIVA', 'POSITIVA', 'NEGATIVA', 'POSITIVA', 'NEGATIVA',
    'POSITIVA', 'NEGATIVA', 'POSITIVA', 'NEGATIVA', 'POSITIVA',
    'NEGATIVA', 'NEGATIVA', 'POSITIVA', 'NEGATIVA', 'POSITIVA'
]

def load_data():
    if os.path.exists(data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['frases'], data['labels']
    else:
        return default_frases.copy(), default_labels.copy()

def save_data(frases, labels):
    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump({'frases': frases, 'labels': labels}, f, ensure_ascii=False)

def train_model(frases, labels):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(frases)
    model = MultinomialNB()
    model.fit(X, labels)
    dump(model, model_file)
    dump(vectorizer, vectorizer_file)
    return model, vectorizer

def load_model():
    model = load(model_file)
    vectorizer = load(vectorizer_file)
    return model, vectorizer

def train_or_load_model(frases, labels):
    if os.path.exists(model_file) and os.path.exists(vectorizer_file):
        print("Carregando modelo e vetorizador existentes...")
        model, vectorizer = load_model()
    else:
        print("Treinando novo modelo...")
        model, vectorizer = train_model(frases, labels)
    X_train = vectorizer.transform(frases)
    predictions = model.predict(X_train)
    accuracy = accuracy_score(labels, predictions)
    print(f"Acurácia no conjunto de treinamento: {accuracy * 100:.2f}%")
    return model, vectorizer

def classify_message(model, vectorizer, message):
    vec = vectorizer.transform([message])
    return model.predict(vec)[0]

def add_new_example(frases, labels, message, correct_label):
    frases.append(message)
    labels.append(correct_label.upper())
    model, vectorizer = train_model(frases, labels)
    save_data(frases, labels)
    print(f"Novo exemplo adicionado e modelo retreinado. Nova acurácia: {accuracy_score(labels, model.predict(vectorizer.transform(frases))) * 100:.2f}%")
    return model, vectorizer

# Carregar dados e modelo
frases, labels = load_data()
model, vectorizer = train_or_load_model(frases, labels)

# Exemplo de classificação
exemplo_mensagens = [
    "O atendimento foi ótimo, gostei muito!",
    "Horrível, nunca mais volto!",
    "O produto é bom mas demorou para chegar"
]

print("\nClassificação dos exemplos:")
for msg in exemplo_mensagens:
    print(f"Mensagem: {msg} -> {classify_message(model, vectorizer, msg)}")

print("\nAgora, digite mensagens para classificar (digite 'sair' para encerrar):")
while True:
    user_input = input("Digite uma mensagem: ").strip()
    if user_input.lower() == 'sair':
        break
    prediction = classify_message(model, vectorizer, user_input)
    print(f"Classificação: {prediction}")
    correct = input("A classificação está correta? (s/n): ").strip().lower()
    if correct == 'n':
        correct_label = input("Qual é o rótulo correto (positiva/negativa)? ").strip().lower()
        if correct_label in ['positiva', 'negativa']:
            model, vectorizer = add_new_example(frases, labels, user_input, correct_label)
        else:
            print("Rótulo inválido. Deve ser 'positiva' ou 'negativa'.")