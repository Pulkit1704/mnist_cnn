FROM continuumio/miniconda3

WORKDIR /app

COPY ./docker_environment.yml /app/

RUN conda env create -f docker_environment.yml && \ 
    echo "source activate pytorch_cnn_project" > ~/.bashrc 

ENV PATH /opt/conda/envs/pytorch_cnn_project/bin:$PATH

COPY . . 

EXPOSE 8080

CMD ["python", "./server.py"]