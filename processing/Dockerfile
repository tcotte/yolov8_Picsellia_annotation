FROM nvcr.io/nvidia/pytorch:24.01-py3
LABEL authors="Tristan_COTTE"

RUN pip install --no-cache-dir ultralytics
RUN pip uninstall opencv -y
RUN pip uninstall opencv-python -y
RUN pip install opencv-python==4.7.0.72

COPY . .

CMD ["yolov8_pre_annotation/launch_processing.py"]
ENTRYPOINT ["python"]
RUN chown -R 42420 ./yolov8_pre_annotation