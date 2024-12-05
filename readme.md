
# Instalação do python
https://medium.com/@francisco_51376/python-wsl-2-75e38ab368c9

sudo apt-get install python3 python3-pip

# Criando alias python = python3
### Para executar o python3 no ubuntu 20.04 deve se digitar python3, mas iremos configurar para que o comando python tambem abra o python3, iremos fazer isso criando um alias no terminal bash.

``` echo “alias python=python3” >> ~/.bashrc   # inseri linha no .bashrc source ~/.bashrc  # recarrega as configurações do .bashrc ```

# Instalação do módulo venv
``` sudo apt install python3-venv ```

# As ambiente virtuais são criadas executando o módulo venv:
``` python3 -m venv env ```

# iniciando o ambiente
``` source env/bin/activate ```

# instalar as dependencias
``` pip install -r requirements.txt ```

# verificar as dependencias
``` pip list ``` 

# Gere o requirements.txt
``` pip freeze > requirements.txt ```

# repositorio remoto set
``` git remote set-url origin git@github.com:usuario/repositorio.git  ```

# verifica o usuario
``` git config --global user.name "Seu Nome" ```
``` git config --global user.email "seu_email@exemplo.com" ```

## Exemplo de Treinamento por Batches com o Hugging Face,
## datasets públicos e prontos para realizar o fine-tuning e treinar o neuralmind/bert-large-portuguese-cased para realizar QeA (Perguntas e Respostas)