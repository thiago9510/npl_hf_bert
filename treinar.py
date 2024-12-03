from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

# Carregar o modelo e tokenizer pré-treinado
model_name = "neuralmind/bert-large-portuguese-cased"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Exemplo de dados de treinamento
data = [
    {"contexto": "O TypeORM é um ORM para Node.js.", "pergunta": "O que é TypeORM?", "respostas": {"texto": ["É um ORM para Node.js"], "answer_start": [3]}},
    {"contexto": "Middleware no Express processa requisições.", "pergunta": "O que é um middleware?", "respostas": {"texto": ["processa requisições"], "answer_start": [13]}}
]

# Converter para Dataset
dataset = Dataset.from_list(data)

# Função para tokenizar os dados
def preprocess(data):
    return tokenizer(data['pergunta'], data['contexto'], truncation=True, padding="max_length", max_length=512)

# Aplicar tokenização
tokenized_data = dataset.map(preprocess, batched=True)

# Definir os argumentos de treinamento
training_args = TrainingArguments(
    output_dir="./chatbot-model",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
    evaluation_strategy="epoch",
)

# Inicializar o Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
)

# Treinar o modelo
trainer.train()

# Salvar o modelo treinado
model.save_pretrained("./chatbot-model")
tokenizer.save_pretrained("./chatbot-model")
