FROM python

WORKDIR /slp

COPY . .

RUN pip install numpy

CMD [ "python", "SLP practice.py" ]

EXPOSE 5000
