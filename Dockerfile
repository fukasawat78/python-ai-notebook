FROM continuumio/anaconda3
WORKDIR /workspace
COPY ./requirements.txt ./workspace/
RUN pip install -r requirements.txt
CMD jupyter-lab --no-browser \
  --port=8888 --ip=0.0.0.0 --allow-root
