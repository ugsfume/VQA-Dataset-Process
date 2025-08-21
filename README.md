# VQA Dataset Process
Batch generation of VQA dataste to be utilised in VLM vertical domain SFT training.

image_augmentor: batch generation of augmented images from template images.

defect_matcher: template match existing defect sample images with template images, extracting corresponding mask images. Extending the VQA dataset.

Also include scripts, mask_generation.py (transform annotation label json to B&W mask images) and qa_gt_generator (create VQA dataset for each augmented sample, programmatically derive ground truth answers from the images).

## How to Run
Entry point is run.py

```
python ./image_augmentor/run.py
```
