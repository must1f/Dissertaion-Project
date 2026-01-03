FROM python:3.13.1

WORKDIR /Jupyter

COPY . /Jupyter

RUN pip install --no-cache-dir -r requirements.txt

RUN pip3 install jupyter

EXPOSE 8888

ENV NAME=world

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]