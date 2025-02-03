import logging
import math
import os
from typing import Union, List, Tuple, Dict

import cv2
import numpy as np
import requests
import torch
import tqdm
from PIL import Image
from picsellia import Asset, DatasetVersion, ModelVersion, Annotation, Label
from picsellia.exceptions import (
    ResourceNotFoundError,
    InsufficientResourcesError,
    PicselliaError,
)
from picsellia.types.enums import InferenceType
from ultralytics import YOLO
from ultralytics.engine.results import Results


class PreAnnotator:
    def __init__(self, client, dataset_version_id, model_version_id, parameters, img_size: int):
        self.model: Union[None, YOLO] = None
        self.client = client
        self.dataset_version: DatasetVersion = client.get_dataset_version_by_id(dataset_version_id)
        self.model_version: ModelVersion = client.get_model_version_by_id(model_version_id)
        self.parameters = parameters

        self.model_labels = []
        self.dataset_labels = []
        self.model_info = {}
        self.model_name = "model-latest"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.inference_size = img_size
        self.max_det = parameters.get("max_det", 300)

        self.dataset_version.delete_all_annotations()

        self._single_class = parameters.get("single_class", False)

        self._labelmap = self._get_labelmap(dataset_version=self.dataset_version)

    def setup_pre_annotation_job(self):
        """
        Set up the pre-annotation job by performing various checks and preparing the model.
        """
        logging.info("Setting up pre-annotation job...")
        self._model_sanity_check()

        if self.dataset_version.type == InferenceType.NOT_CONFIGURED:
            self._set_dataset_type_to_model_type()
            self._create_labels_in_dataset()

        # else:
        #     self._type_coherence_check()
        #     self._labels_coherence_check()

        self._download_model_weights()
        self._load_yolov8_model()

    def pre_annotate(self, confidence_threshold: float = 0.25):
        """
        Processes and annotates assets in the dataset using the YOLOv8 model.

        Args:
            confidence_threshold (float, optional): A threshold value used to filter
                                                    the bounding boxes based on their
                                                    confidence scores. Only boxes with
                                                    confidence scores above this threshold
                                                    are annotated. Defaults to 0.5.
        """
        dataset_size = self.dataset_version.sync()["size"]
        batch_size = self.parameters.get("batch_size", 4)
        iou_threshold = self.parameters.get("iou", 0.7)
        # confidence_threshold = self.parameters.get("confidence", 0.25)
        batch_size = min(batch_size, dataset_size)

        total_batch_number = math.ceil(dataset_size / batch_size)

        logging.info(
            f"\n-- Starting processing {total_batch_number} batch(es) of {batch_size} image(s) | "
            f"Total images: {dataset_size} --"
        )

        for batch_number in tqdm.tqdm(
                range(total_batch_number),
                desc="Processing batches",
                unit="batch",
        ):
            assets = self.dataset_version.list_assets(limit=batch_size, offset=batch_number * batch_size)
            url_list = [asset.sync()["data"]["presigned_url"] for asset in assets]
            predictions = self.model(url_list,
                                     iou=iou_threshold,
                                     imgsz=self.inference_size,
                                     max_det=self.max_det,
                                     conf=confidence_threshold)

            for asset, prediction in list(zip(assets, predictions)):
                if len(asset.list_annotations()) == 0:
                    if len(prediction) > 0:
                        if self.dataset_version.type == InferenceType.OBJECT_DETECTION:
                            self._format_and_save_rectangles(asset, prediction, confidence_threshold)

    def _get_label_by_name(self, labelmap: Dict[str, Label], label_name: str) -> Label:
        if label_name not in labelmap:
            raise ValueError(f"The label {label_name} does not exist in the labelmap.")

        return labelmap[label_name]

    def _get_labelmap(self, dataset_version: DatasetVersion) -> Dict[str, Label]:
        return {label.name: label for label in dataset_version.list_labels()}

    @staticmethod
    def _reset_annotations(asset) -> None:
        """
        Erase current annotations of an asset sent as parameter.
        :param asset: asset in which the annotations will be removed
        """
        if asset.list_annotations() != []:
            # update without any annotation
            asset.get_annotation().overwrite()

    def _format_and_save_rectangles(self, asset: Asset, prediction: Results,
                                    confidence_threshold: float = 0.25) -> None:
        # remove current annotations
        self._reset_annotations(asset)

        boxes = prediction.boxes.xyxyn.cpu().numpy()
        scores = prediction.boxes.conf.cpu().numpy()
        labels = prediction.boxes.cls.cpu().numpy().astype(np.int16)

        #  Convert predictions to Picsellia format
        rectangle_list: list = []
        nb_box_limit = self.max_det
        if len(boxes) < nb_box_limit:
            nb_box_limit = len(boxes)
        if len(boxes) > 0:
            annotation: Annotation = asset.create_annotation(duration=0.0)
        else:
            return
        for i in range(nb_box_limit):
            if scores[i] >= confidence_threshold:
                try:
                    if not self._single_class:
                        label = self._get_label_by_name(labelmap=self._labelmap,
                                                        label_name=prediction.names[labels[i]])
                    else:
                        label = next(iter(self._labelmap.values()))
                    e = boxes[i].tolist()
                    box = [
                        int(e[0] * asset.width),
                        int(e[1] * asset.height),
                        int((e[2] - e[0]) * asset.width),
                        int((e[3] - e[1]) * asset.height),
                    ]
                    box.append(label)
                    rectangle_list.append(tuple(box))
                except ResourceNotFoundError as e:
                    print(e)
                    continue

        if len(rectangle_list) > 0:
            annotation.create_multiple_rectangles(rectangle_list)
            logging.info(f"Asset: {asset.filename} pre-annotated.")

    @staticmethod
    def round_to_multiple(number: int, multiple: int) -> int:
        """
        Round some number to it's nearest multiple.
        Ex : number 99 / multiple 25 -> return 100
        """
        return multiple * round(number / multiple)

    def _model_sanity_check(self) -> None:
        """
        Perform a sanity check on the model.
        """
        self._check_model_file_integrity()
        self._validate_model_inference_type()
        logging.info(f"Model {self.model_version.name} passed sanity checks.")

    def _check_model_file_integrity(self) -> None:
        """
        Check the integrity of the model file by verifying it exists as "model-latest" and is an ONNX model.

        Raises:
            ResourceNotFoundError: If the model file is not an ONNX file.
        """
        model_file = self.model_version.get_file(self.model_name)
        if not model_file.filename.endswith(".pt") and not model_file.filename.endswith(".pth"):
            raise ResourceNotFoundError("Model file must be a pth file.")

    def _validate_model_inference_type(self) -> None:
        """
        Validate the model's inference type.

        Raises:
            PicselliaError: If the model type is not configured.
        """
        if self.model_version.type == InferenceType.NOT_CONFIGURED:
            raise PicselliaError("Model type is not configured.")

    def _set_dataset_type_to_model_type(self) -> None:
        """
        Set the dataset type to the model type.
        """
        self.dataset_version.set_type(self.model_version.type)
        logging.info(f"Dataset type set to {self.model_version.type}")

    def _create_labels_in_dataset(self) -> None:
        """
        Creates labels in the dataset based on the model's labels. It first retrieves the model's labels,
        then creates corresponding labels in the dataset version if they do not already exist.

        This method updates the 'model_labels' and 'dataset_labels' attributes of the class with the
        labels from the model (if they don't already exist) and the labels currently in the dataset, respectively.
        """
        if not self.model_labels:
            self.model_labels = self._get_model_labels()

        for label in tqdm.tqdm(self.model_labels):
            self.dataset_version.create_label(name=label)

        self.dataset_labels = [
            label.name for label in self.dataset_version.list_labels()
        ]
        logging.info(f"Labels created in dataset: {self.dataset_labels}")

    def _validate_dataset_and_model_type(self) -> None:
        """
        Validate that the dataset type matches the model type.

        Raises:
            PicselliaError: If the dataset type does not match the model type.
        """
        if self.dataset_version.type != self.model_version.type:
            raise PicselliaError(
                f"Dataset type {self.dataset_version.type} does not match model type {self.model_version.type}."
            )

    def _validate_label_overlap(self) -> None:
        """
        Validate that there is an overlap between model labels and dataset labels.

        Raises:
            PicselliaError: If no overlapping labels are found.
        """
        self.model_labels = self._get_model_labels()
        self.dataset_labels = [
            label.name for label in self.dataset_version.list_labels()
        ]

        model_labels_set = set(self.model_labels)
        dataset_labels_set = set(self.dataset_labels)

        overlapping_labels = model_labels_set.intersection(dataset_labels_set)
        non_overlapping_dataset_labels = dataset_labels_set - model_labels_set

        if not overlapping_labels:
            raise PicselliaError(
                "No overlapping labels found between model and dataset. "
                "Please check the labels between your dataset version and your model.\n"
                f"Dataset labels: {self.dataset_labels}\n"
                f"Model labels: {self.model_labels}"
            )

        # Log the overlapping labels
        logging.info(f"Using labels: {list(overlapping_labels)}")

        if non_overlapping_dataset_labels:
            logging.info(
                f"WARNING: Some dataset version's labels are not present in model "
                f"and will be skipped: {list(non_overlapping_dataset_labels)}"
            )

    def _download_model_weights(self) -> None:
        """
        Download the model weights and save it in `self.model_weights_path`.
        """
        model_weights = self.model_version.get_file(self.model_name)
        model_weights.download()
        self.model_weights_path = os.path.join(os.getcwd(), model_weights.filename)
        logging.info(f"Model weights {model_weights.filename} downloaded successfully")

    def _load_yolov8_model(self) -> None:
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_weights_path)
            logging.info("Model loaded in memory.")
        except Exception as e:
            raise PicselliaError(f"Impossible to load saved model located at: {self.model_weights_path}")

    def _get_model_labels(self) -> List[str]:
        """
        Get the labels from the model.

        Returns:
            list[str]: A list of label names from the model.
        Raises:
            InsufficientResourcesError: If no labels are found or if labels are not in dictionary format.
        """
        self.model_info = self.model_version.sync()
        if "labels" not in self.model_info:
            raise InsufficientResourcesError(
                f"No labels found for model {self.model_version.name}."
            )

        if not isinstance(self.model_info["labels"], dict):
            raise InsufficientResourcesError(
                "Model labels must be in dictionary format."
            )

        return list(self.model_info["labels"].values())
