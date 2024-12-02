from transformers import pipeline

# Carregar o pipeline de pergunta e resposta  neuralmind/bert-large-portuguese-cased
qa_pipeline = pipeline("question-answering")

# Testar o modelo
result = qa_pipeline({
    'context': "Hugging Face é uma empresa focada em criar ferramentas de aprendizado de máquina para desenvolvedores.",
    'question': "O que é Hugging Face?"
})

print(result)
