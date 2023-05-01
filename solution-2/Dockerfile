FROM conda/miniconda3

WORKDIR /app
COPY . /app

# RUN conda env create -f environment.yml && \
#     conda activate torch

RUN conda env create -f environment.yml
RUN conda activate torch

EXPOSE 5000

CMD ["python", "app.py"]