#FROM python
#COPY . /usr/app/
#EXPOSE 5000
#WORKDIR /usr/app/
#RUN pip install -r requirements.txt
#CMD python form_enduser.py


FROM python
WORKDIR /telecomfiles
COPY requirements.txt /telecomfiles
EXPOSE 8000
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements.txt
COPY . /telecomfiles
ENTRYPOINT ["python3"]
CMD ["form_enduser.py"]