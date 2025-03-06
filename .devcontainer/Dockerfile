FROM python:3.12-slim AS build

# Stel de werkdirectory in
WORKDIR /code

# Kopieer alle files naar de werkdirectory
COPY .github/ /code/.github/
COPY .vscode/ /code/.vscode/
COPY data/ /code/data/
COPY docs/ /code/docs/
COPY src/ /code/src/
COPY .gitattributes /code/.gitattributes
COPY .gitignore /code/.gitignore
COPY CITATION.cff /code/CITATION.cff
COPY LICENSE /code/LICENSE
COPY pixi.lock /code/pixi.lock
COPY pixi.toml /code/pixi.toml
COPY README.md /code/README.md
COPY test.py /code/test.py

# RUN pip install -r /code/requirements.txt
RUN apt-get -y update
RUN apt-get -y install git curl
RUN curl -fsSL https://pixi.sh/install.sh | bash
# RUN pixi install
