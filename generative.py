from transformers import pipeline

# Configurando o pipeline de geração de texto
text_generation_pipeline = pipeline(
    "text-generation",
    model="gpt2",
    device=0,
    pad_token_id=50256  # Define o token de preenchimento
)

# Contexto
contexto = (
"""
A pandemia de COVID-19, também conhecida como pandemia de coronavírus, é uma pandemia em curso de COVID-19, 
uma doença respiratória causada pelo coronavírus da síndrome respiratória aguda grave 2 (SARS-CoV-2). 
O vírus tem origem zoonótica e o primeiro caso conhecido da doença remonta a dezembro de 2019 em Wuhan, na China. 
Em 20 de janeiro de 2020, a Organização Mundial da Saúde (OMS) classificou o surto 
como Emergência de Saúde Pública de Âmbito Internacional e, em 11 de março de 2020, como pandemia. 
Em 18 de junho de 2021, 177 349 274 casos foram confirmados em 192 países e territórios, 
com 3 840 181 mortes atribuídas à doença, tornando-se uma das pandemias mais mortais da história.
Os sintomas de COVID-19 são altamente variáveis, variando de nenhum a doenças com risco de morte. 
O vírus se espalha principalmente pelo ar quando as pessoas estão perto umas das outras. 
Ele deixa uma pessoa infectada quando ela respira, tosse, espirra ou fala e entra em outra pessoa pela boca, nariz ou olhos.
Ele também pode se espalhar através de superfícies contaminadas. 
As pessoas permanecem contagiosas por até duas semanas e podem espalhar o vírus mesmo se forem assintomáticas.
"""
)

# Prompt ajustado para solicitar um resumo sem incluir o próprio texto
prompt = (
    f"""Contexto: {contexto}, Sabendo disso gere um texto falando sobre a importante de se cuidar
    """
)

# Gerando a resposta
resposta_detalhada_pipeline = text_generation_pipeline(
    prompt,
    max_new_tokens=500,  # Limite de tokens novos gerados
    num_return_sequences=1,  # Apenas uma resposta
    temperature=0.7,  # Controle de criatividade
    top_p=0.9  # Probabilidade cumulativa
)[0]["generated_text"]

# Exibindo a resposta
print("Resposta gerada: " + resposta_detalhada_pipeline)
