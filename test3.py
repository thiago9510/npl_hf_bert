import transformers
from transformers import pipeline

# source: https://pt.wikipedia.org/wiki/Pandemia_de_COVID-19
context = """
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

model_name = 'pierreguillou/bert-large-cased-squad-v1.1-portuguese'
nlp = pipeline("question-answering", model=model_name, tokenizer="pierreguillou/bert-large-cased-squad-v1.1-portuguese", device=0)

question = "Quando começou a pandemia de Covid-19 no mundo?"

result = nlp(question=question, context=context)

print(f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")

# Answer: 'dezembro de 2019', score: 0.5087, start: 290, end: 306
