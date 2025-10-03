# Documenta-o-do-Prot-tipo-de-An-lise-de-Sentimentos-em-Python

Protótipo de Análise de Sentimentos com IA em Python
Descrição
Este é um protótipo simples de Inteligência Artificial (IA) para análise de sentimentos em mensagens de clientes, desenvolvido em Python. O sistema classifica textos como POSITIVA ou NEGATIVA usando Machine Learning (ML) com o algoritmo Naive Bayes da biblioteca scikit-learn. Baseado no desafio "desafio_IA_Python.docx", o código treina um modelo com um conjunto pequeno de frases, permite classificação de novas mensagens e inclui extras como salvamento de modelo e aprendizado incremental.
Ideal para demonstrar habilidades em Python e ML em contextos como startups ou análise de feedbacks.
Funcionalidades

Treinamento de modelo com dados iniciais (20 frases em português).
Classificação interativa de mensagens via console.
Persistência: Salva modelo (joblib) e dados (JSON) para reutilização.
Aprendizado: Adiciona novas frases com correções e retreina o modelo.
Métricas: Exibe acurácia no conjunto de treinamento.

Requisitos

Python 3.x
Bibliotecas: scikit-learn, joblib

Instalação:
textpip install scikit-learn joblib
Como Usar

Clone o repositório ou copie o código para um arquivo sentiment_analysis.py.
Execute o script:
textpython sentiment_analysis.py

O sistema treina/carrega o modelo e classifica exemplos iniciais.
Digite mensagens para classificar (ex: "Adorei o produto!").
Se a classificação estiver errada, corrija para adicionar ao modelo.
Digite 'sair' para encerrar.

Exemplo de saída:
textAcurácia no conjunto de treinamento: 100.00%
Classificação dos exemplos:
Mensagem: O atendimento foi ótimo, gostei muito! -> POSITIVA
...
Digite uma mensagem: Horrível serviço.
Classificação: NEGATIVA
Estrutura do Código

load_data() e save_data(): Gerenciam dados em JSON.
train_model() e load_model(): Treinam e carregam o modelo.
classify_message(): Classifica uma mensagem.
add_new_example(): Adiciona e retreina com novos dados.
Loop principal: Interação com usuário.

Limitações e Melhorias

Limitações: Dados pequenos podem causar erros em frases complexas; sem pré-processamento avançado.
Sugestões:

Adicionar NLTK para stemming e remoção de stop words.
Usar TF-IDF para melhor vetorização.
Integrar datasets maiores (ex: Kaggle).
Criar API com Flask para uso web.



Contribuição
Sinta-se à vontade para fork e pull requests! Sugestões são bem-vindas.
Licença
MIT License - Livre para uso e modificação.
