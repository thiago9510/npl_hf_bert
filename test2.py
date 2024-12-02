from transformers import pipeline 

# Carregando o modelo e o tokenizador neuralmind/bert-large-portuguese-cased / neuralmind/bert-large-portuguese-cased-squad
qa_pipeline = pipeline("question-answering", model="neuralmind/bert-large-portuguese-cased", device=0)

# Contexto e pergunta
context = "Python é uma linguagem de programação de alto nível, muito usada em ciência de dados, inteligência artificial e automação de scripts. Criada por Guido van Rossum, ela é fácil de aprender, o que o torna uma excelente escolha para iniciantes."

question = "Quem criou o Python?"

# Resposta
response = qa_pipeline({
    'context': ( 'gere um texto detlhado com base nas informações a seguir:' + context),
    'question': question
})

print("Resposta:", response['answer'])
