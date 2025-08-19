# image-augmentor
Batch generation of augmented images to be utilised in VLM vertical domain SFT training.
Also include scripts, mask_generation.py (transform annotation label json to B&W mask images) and qa_gt_generator (create VQA dataset for each augmented sample, programmatically derive ground truth answers from the images).

## How to Run
Entry point is run.py

```
python ./image_augmentor/run.py
```
