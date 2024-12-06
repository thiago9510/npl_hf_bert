from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Carregar o modelo GPT-2 para português
model_name = "neuralmind/bert-large-portuguese-cased"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Criar um pipeline de geração de texto
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Gerar texto
prompt = "A inteligência artificial no Brasil"
result = generator(prompt, max_length=100, num_return_sequences=1)

print(result[0]['generated_text'])