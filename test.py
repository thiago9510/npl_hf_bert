# Importação da biblioteca transformers para usar o modelo pré-treinado
from transformers import pipeline

# Carregar o pipeline de pergunta e resposta
qa_pipeline = pipeline(
    "question-answering",
    #model="neuralmind/bert-large-portuguese-cased",
    tokenizer="pierreguillou/bert-large-cased-squad-v1.1-portuguese",
    model="pierreguillou/bert-large-cased-squad-v1.1-portuguese",
    #tokenizer="gpt2,
    device=0
)

# Banco de contextos para definir o que será usado na resposta
context_database = [
    {
        "input": ["Hugging Face", "empresa", "ferramentas de aprendizado de máquina"],
        "contexto": "Hugging Face é uma empresa focada em criar ferramentas de aprendizado de máquina para desenvolvedores.",
    },
    {
        "input": ["Python", "linguagem de programação", "coding"],
        "contexto": "Python é uma linguagem de programação versátil, usada para desenvolvimento web, ciência de dados, automação e muito mais.",
    },
    {
        "input": ["transformers", "modelo pré-treinado", "NLP"],
        "contexto": "Transformers são modelos de aprendizado profundo desenvolvidos para processar linguagem natural. Exemplos incluem BERT, GPT e outros.",
    }
]

# Função para selecionar o contexto com base na pergunta
def selecionar_contexto(pergunta):
    for item in context_database:
        if any(keyword.lower() in pergunta.lower() for keyword in item["input"]):
            return item["contexto"]
    return "Desculpe, não encontrei informações suficientes para responder a sua pergunta."

# Função para responder a pergunta
def responder_pergunta(pergunta):
    contexto = selecionar_contexto(pergunta)
    if "Desculpe" in contexto:
        return contexto  # Retorna uma mensagem padrão se o contexto não for encontrado
    resultado = qa_pipeline({"question": pergunta, "context": contexto})
    return resultado["answer"]

# Interface de interação
print("Chatbot ativado! Faça sua pergunta:")
while True:
    pergunta = input("Você: ")
    if pergunta.lower() in ["sair", "exit", "tchau"]:
        print("Chatbot: Até mais!")
        break

    resposta = responder_pergunta(pergunta)
    print(f"Chatbot: {resposta}")
