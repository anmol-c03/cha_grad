FROM python:3.10

RUN mkdir -p /home/cha_grad

COPY ./src /home/cha_grad

COPY ./notebook /home/cha_grad

COPY ./requirements.txt /home/cha_grad

WORKDIR /home/cha_grad

RUN apt update -y && \
    apt install -y nano && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

CMD ["python3","scratch.py"] 
# if u want to train your own model, create a new .py script (say new.py ) in src and replace 
# CMD ["python3","scratch.py"] with CMD ["python3","new.py"] 
