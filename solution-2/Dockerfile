FROM continuumio/miniconda3

WORKDIR /app
COPY . /app

RUN conda env create -f environment.yml && \
    conda clean --all --yes

SHELL ["/bin/bash", "-c"]

CMD ["bash", "-c", "source activate torch && python app.py"]