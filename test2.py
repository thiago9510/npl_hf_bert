from transformers import pipeline

# Configurando o pipeline de QA
qa_pipeline = pipeline(
    "question-answering",
    #model="neuralmind/bert-large-portuguese-cased",
    #tokenizer="neuralmind/bert-large-portuguese-cased",
    model="pierreguillou/bert-large-cased-squad-v1.1-portuguese",
    tokenizer="pierreguillou/bert-large-cased-squad-v1.1-portuguese",
    device=0
)

# Contexto e pergunta
contexto = (
    "João é engenheiro de software e trabalha em uma empresa de tecnologia há cinco anos. "
    "Ele é especializado em desenvolvimento de sistemas back-end e está atualmente aprendendo "
    "sobre machine learning. No tempo livre, João gosta de jogar futebol e estudar inteligência "
    "artificial. Ele também é voluntário em projetos que promovem a inclusão digital em comunidades carentes."
)

pergunta = "do que ele gosta alé, de jogar futebol ?"

# Fazendo a predição
resultado = qa_pipeline(question=pergunta, context=contexto)

# Exibindo o resultado
print("Resposta:", resultado["answer"])
print("Confiança:", resultado["score"])
