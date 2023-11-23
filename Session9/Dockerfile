FROM agrigorev/zoomcamp-bees-wasps:v2

RUN pip install numpy pillow tensorflow==2.14.0 bytesbufio
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl

COPY lambda_code.py .

CMD [ "lambda_code.lambda_handler" ]