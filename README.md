📊 Protótipo de Análise de Sentimentos em Python.



📝 Descrição:

Protótipo de IA para classificar mensagens como POSITIVA ou NEGATIVA usando ML (Naive Bayes). Treina com 20 frases iniciais, classifica interativamente, salva modelo/dados e aprende com correções. Baseado no desafio "desafio_IA_Python.docx". Ideal para feedbacks de clientes! 🚀


✨ Funcionalidades:

🧠 Treinamento com dados iniciais e acurácia exibida.

🔍 Classificação de novas mensagens via console.

💾 Persistência em joblib (modelo) e JSON (dados).

📈 Aprendizado incremental: adicione e retreine.

📊 Métricas: acurácia no treino (100% inicial).


🛠️ Requisitos:

Python 3.x
Bibliotecas: scikit-learn, joblib

Instalação:

textpip install scikit-learn joblib


🚀 Como Usar:

Copie o código para analise_sentimentos.py.
Execute: python analise_sentimentos.py.
Veja exemplos classificados.
Digite mensagens (ex: "Adorei!").
Corrija erros para melhorar o modelo.
Saia com 'sair'. 🔄

Exemplo:
textAcurácia: 100%  
Mensagem: Ótimo! → POSITIVA


🗂️ Estrutura do Código:

load_data() / save_data(): Gerencia JSON.
train_model() / load_model(): Treina/carrega ML.
classify_message(): Classifica texto.
add_new_example(): Adiciona e retreina.
Loop: Interação usuário.


⚠️ Limitações e Melhorias:

Limitações: Dados pequenos causam erros em frases complexas. 😕
Melhorias:

➕ NLTK para pré-processamento (stop words, stemming).
🔄 TF-IDF para vetorização melhor.
📚 Datasets maiores (Kaggle).
🌐 API com Flask.



🤝 Contribuição:
Fork e envie pull requests! Ideias bem-vindas. ⭐
📜 Licença:
MIT - Livre para uso/modificação. © 2023.
