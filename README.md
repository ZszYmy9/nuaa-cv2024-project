# nuaa-cv2024-project
南京航空航天大学2024年计算机视觉课程实践作业
## 小组成员
162100101 郑思哲

162100103 周健文

162100104 周扬超

## 指导教师
梁栋

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

Our project takes Cityscapes val set for testing.You need to download the [Cityscapes](https://www.cityscapes-dataset.com/)datasets.

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
### DeeplabV3+
The pretrained model can be accessed [here](https://drive.google.com/file/d/1t7TC8mxQaFECt4jutdq_NMnWxdm6B-Nb/view?usp=sharing).
You can have a test on val set by running the command:
```bash
python main.py --model deeplabv3plus_resnet101 --gpu_id 0 --ckpt checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth --test_only --save_val_results
```
The results will be saved at ./results.

### HRNetV2
The pretrained model can be accessed [Github](https://github.com/hsfzxjy/models.storage/releases/download/HRNet-OCR/hrnet_cs_8090_torch11.pth)/[BaiduYun(Access Code:pmix)](https://pan.baidu.com/s/1KyiOUOR0SYxKtJfIlD5o-w).
You can have a test on val set by running the command:
```bash
bash run_without_ocr.sh
```
The results will be saved at test_results.

### HRNetV2 + OCR
The pretrained model can be accessed The pretrained model can be accessed [Github](https://github.com/hsfzxjy/models.storage/releases/download/HRNet-OCR/hrnet_cs_8090_torch11.pth)/[BaiduYun(Access Code:pmix)](https://pan.baidu.com/s/1KyiOUOR0SYxKtJfIlD5o-w).
You can have a test on val set by running the command:
```bash
bash run_without_ocr.sh
```
The results will be saved at test_results.
You can have a test on val set by running the command:
```bash
bash run_with_ocr.sh
```
The results will be saved at test_results.

## 风格迁移
### StyTr^2
For training StyTr2, your directory tree should be look like this:
````bash
$SEG_ROOT/data
├── train2014
│   ├── 1.png
│   ├── 2.png
│   ├── ...
├── wikiart
│   ├── 1.png
│   ├── 2.png
│   ├── ...
````
If you want to train your StyTr^2, you can run the command:
````bash
python train.py --batch_size 8
````
The pretrained model can be accessed [here](https://github.com/diyiiyiii/StyTR-2)

If you want to have a test, you can run the command:
````bash
python test.py
````
### PuffNet
Your directory tree should be the same as the above.

If you want to train your PuffNet on a file folder, you can run the command:
````bash
python train.py --batch_size 8 --content_data <content_data> --style_data <style_data> --train True
````

If you want to test your PuffNet on a file folder, you can run the command:
````bash
python train.py --content_data <content_data> --style_data <style_data> --train False
````

If you want to test your PuffNet on a couple of images, you can run the command:
````bash
python test.py
````

### Dreambooth
For training Dreambooth, your directory tree should be look like this:
````bash
$SEG_ROOT/A
├── 00
│   ├── images
│   │   ├── cow.png
│   │   ├── ...
│   └──prompt.json
├── 01
│   ├── images
│   │   ├── dog.png
│   │   ├── ...
│   └──prompt.json
├── ...
````
If you want to train your Dreambooth, you can run the command:
````bash
bash train_all.sh
````
If you want to test your Dreambooth, you can run the command:
````bash
python run_all.py
````



