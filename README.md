<div align="center">

<h1>
    <br> 
    Self-supervised ControlNet with Spatio-Temporal Mamba for Real-world Video Super-resolution(CVPR2025)
</h1>

Shijun Shi<sup>1*</sup>, Jing Xu<sup>2*</sup>, Lijing Lu<sup>3&dagger;</sup>, Zhihang Li<sup>4&dagger;</sup>, Kai Hu<sup>1</sup>

_<sup>1</sup>Jiangnan University_  
_<sup>2</sup>University of Science and Technology of China_  
_<sup>3</sup>Peking University_ 
_<sup>4</sup>Chinese Academy of Sciences_  

<div>
    <h4 align="center">
        <a href="https://ssj9596.github.io/scst-project/" target='_blank'>
        <img src="https://img.shields.io/badge/🐳-Project%20Page-blue">
        </a>
    </h4>
</div>

</div>


## 🔥 Update
- [2025.03] Inference code and checkpoint is released.


## 📈 Our model can do both ISR and VSR. Hope you can enjoy it.
### Realistic Image SR
<img src="assets/img3.gif" width="390px"/> <img src="assets/img3.gif" width="390px"/>

### Realistic Video SR
<img src="assets/vid3.gif" width="390px"/> <img src="assets/vid3.gif" width="390px"/>

## 🎬 Overview
![overall_structure](assets/pipeline.png)

## 🔧 Dependencies and Installation
1. Clone Repo
    ```bash
    git clone https://github.com/sczhou/Upscale-A-Video.git
    cd Upscale-A-Video
    ```

2. Create Conda Environment and Install Dependencies
    ```bash
    # create new conda env
    conda create -n UAV python=3.9 -y
    conda activate UAV

    # install python dependencies
    pip install -r requirements.txt
    ```

3. Download Models

   (a) Download pretrained models and configs from [Google Drive](https://drive.google.com/drive/folders/1O8pbeR1hsRlFUU8O4EULe-lOKNGEWZl1?usp=sharing) and put them under the `pretrained_models/upscale_a_video` folder.

   The [`pretrained_models`](./pretrained_models) directory structure should be arranged as:

    ```
    ├── pretrained_models
    │   ├── upscale_a_video
    │   │   ├── low_res_scheduler
    │   │       ├── ...
    │   │   ├── propagator
    │   │       ├── ...
    │   │   ├── scheduler
    │   │       ├── ...
    │   │   ├── text_encoder
    │   │       ├── ...
    │   │   ├── tokenizer
    │   │       ├── ...
    │   │   ├── unet
    │   │       ├── ...
    │   │   ├── vae
    │   │       ├── ...
    ```
    
    (a) (Optional) LLaVA can be downloaded automatically when set `--use_llava` to `True`, for users with access to huggingface.


## ☕️ Quick Inference

The `--input_path` can be either the path to a single video or a folder containing multiple videos.

We provide several examples in the [`inputs`](./inputs) folder. 
Run the following commands to try it out:

```shell
## AIGC videos
python inference_upscale_a_video.py \
-i ./inputs/aigc_1.mp4 -o ./results -n 150 -g 6 -s 30 -p 24,26,28

python inference_upscale_a_video.py \
-i ./inputs/aigc_2.mp4 -o ./results -n 150 -g 6 -s 30 -p 24,26,28

python inference_upscale_a_video.py \
-i ./inputs/aigc_3.mp4 -o ./results -n 150 -g 6 -s 30 -p 20,22,24
```

```shell
## old videos/movies/animations 
python inference_upscale_a_video.py \
-i ./inputs/old_video_1.mp4 -o ./results -n 150 -g 9 -s 30

python inference_upscale_a_video.py \
-i ./inputs/old_movie_1.mp4 -o ./results -n 100 -g 5 -s 20 -p 17,18,19

python inference_upscale_a_video.py \
-i ./inputs/old_movie_2.mp4 -o ./results -n 120 -g 6 -s 30 -p 8,10,12

python inference_upscale_a_video.py \
-i ./inputs/old_animation_1.mp4 -o ./results -n 120 -g 6 -s 20 --use_video_vae
```

If you notice any color discrepancies between the output and the input, you can set `--color_fix` to `"AdaIn"` or `"Wavelet"`. By default, it is set to `"None"`.



## 🎞️ YouHQ Dataset
The datasets are hosted on Google Drive

| Dataset | Link | Description|
| :----- | :--: | :---- | 
| YouHQ-Train | [Google Drive](https://drive.google.com/file/d/1f8g8gTHzQq-cKt4s94YQXDwJcdjL59lK/view?usp=sharing)| 38,576 videos for training, each of which has around 32 frames.|
| YouHQ40-Test| [Google Drive](https://drive.google.com/file/d/1rkeBQJMqnRTRDtyLyse4k6Vg2TilvTKC/view?usp=sharing) | 40 video clips for evaluation, each of which has around 32 frames.|

## 📑 Citation

   If you find our repo useful for your research, please consider citing our paper:

   ```bibtex
   @inproceedings{zhou2024upscaleavideo,
      title={{Upscale-A-Video}: Temporal-Consistent Diffusion Model for Real-World Video Super-Resolution},
      author={Zhou, Shangchen and Yang, Peiqing and Wang, Jianyi and Luo, Yihang and Loy, Chen Change},
      booktitle={CVPR},
      year={2024}
   }
   ```


## 📝 License

This project is licensed under <a rel="license" href="./LICENSE">NTU S-Lab License 1.0</a>. Redistribution and use should follow this license.


## 📧 Contact
If you have any questions, please feel free to reach us at `shangchenzhou@gmail.com` or `peiqingyang99@outlook.com`. 
