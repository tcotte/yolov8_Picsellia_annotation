# YOLOv8 annotator

This processing works only if the *single_clas* parameter is set to True.

TODO: add multi class possibility to this pre-annotator. To do that, integrate validation functions in 
*setup_pre_annotation_job*: *_type_coherence_check* and *self._labels_coherence_check*

### Parameters

List of parameters:

- confidence_threshold: float
- image_size: int
- max_det: int
- single_class: str
- batch_size: int
- iou: float
