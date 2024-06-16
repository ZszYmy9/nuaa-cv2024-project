# nuaa-cv2024-project
南京航空航天大学2024年计算机视觉课程实践作业
## 小组成员
162100101 郑思哲

162100103 周健文

162100104 周扬超

## 底层视觉低层视觉及传统方法

### A Single Image Haze Removal Using Dark Channel Prior
You can remove a single image's haze by running the command:

```bash
python A/dehaze.py --img_roor <img_root> --output <output_root>
```

You can also modify the hyperparameter configuartion.

### Background Substraction With Gaussian Mixture Model
You can achieve a background substraction for an image floder by running the command:

```bash
python A/gmm.py --input <input_root>  --output <output_root>
```

You can also modify the hyperparameter configuartion.

## 高层视觉语义分割

### Date Preparation

You need to download the [Cityscapes](https://www.cityscapes-dataset.com/)datasets.

Your directory tree should be look like this:
````bash
$SEG_ROOT/data
├── cityscapes
│   ├── gtFine
│   │   ├── test
│   │   ├── train
│   │   └── val
│   └── leftImg8bit
│       ├── test
│       ├── train
│       └── val
````
