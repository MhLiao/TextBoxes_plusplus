# TextBoxes++: A Single-Shot Oriented Scene Text Detector

### Introduction
This is an application for scene text detection (TextBoxes++) and recognition (CRNN).

TextBoxes++ is a unified framework for oriented scene text detection with a single network. It is an extended work of [TextBoxes](https://github.com/MhLiao/TextBoxes). [CRNN](https://github.com/bgshih/crnn) is an open-source text recognizer. 
The code of TextBoxes++ is based on [SSD](https://github.com/weiliu89/caffe/tree/ssd) and [TextBoxes](https://github.com/MhLiao/TextBoxes). The code of CRNN is modified from [CRNN](https://github.com/bgshih/crnn).


For more details, please refer to our [arXiv paper](https://arxiv.org/abs/1801.02765). 
### Citing the related works

Please cite the related works in your publications if it helps your research:

    @article{Liao2018Text,
      title = {{TextBoxes++}: A Single-Shot Oriented Scene Text Detector},
      author = {Minghui Liao, Baoguang Shi and Xiang Bai},
      journal = {{IEEE} Transactions on Image Processing},
      doi  = {10.1109/TIP.2018.2825107},
      url = {https://doi.org/10.1109/TIP.2018.2825107},
      volume = {27},
      number = {8},
      pages = {3676--3690},
      year = {2018}
    }
    
    @inproceedings{LiaoSBWL17,
      author    = {Minghui Liao and
                   Baoguang Shi and
                   Xiang Bai and
                   Xinggang Wang and
                   Wenyu Liu},
      title     = {TextBoxes: {A} Fast Text Detector with a Single Deep Neural Network},
      booktitle = {AAAI},
      year      = {2017}
    }
    
    @article{ShiBY17,
      author    = {Baoguang Shi and
                   Xiang Bai and
                   Cong Yao},
      title     = {An End-to-End Trainable Neural Network for Image-Based Sequence Recognition
                   and Its Application to Scene Text Recognition},
      journal   = {{IEEE} TPAMI},
      volume    = {39},
      number    = {11},
      pages     = {2298--2304},
      year      = {2017}
    }

### Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Docker](#docker)
4. [Models](#models)
5. [Demo](#demo)
6. [Train](#train)


### Requirements

**NOTE** There is partial support for a docker image. See `docker/README.md`. (Thank you for the PR from [@mdbenito](https://github.com/mdbenito))

    Torch7 for CRNN; 
    g++-5; cuda8.0; cudnn V5.1 (cudnn 6 and cudnn 7 may fail); opencv3.0
  
Please refer to [Caffe Installation](http://caffe.berkeleyvision.org/install_apt.html) to ensure other dependencies;


### Installation

1. compile TextBoxes++ (This is a modified version of caffe so you do not need to install the official caffe)
  ```Shell
  # Modify Makefile.config according to your Caffe installation.
  cp Makefile.config.example Makefile.config
  make -j8
  # Make sure to include $CAFFE_ROOT/python to your PYTHONPATH.
  make py
  ```
2. compile CRNN (Please refer to [CRNN](https://github.com/bgshih/crnn) if you have trouble with the compilation.)
  ```Shell
  cd crnn/src/
  sh build_cpp.sh
  ```

### Docker
(Thanks for the PR from [@idotobi](https://github.com/idotobi))

Build Docke Image

    docker build -t tbpp_crnn:gpu .

This can take +1h, so go get a coffee ;)

Once this is done you can start a container via `nvidia-docker`. 

    nvidia-docker run -it --rm tbpp_crnn:gpu bash

To check if the GPU is available inside the docker container you can run `nvidia-smi`.

It's recommendable to mount the `./models` and `./crnn/model/` directories to include the downloaded [models](#models).

    nvidia-docker run -it \
                      --rm \
                      -v ${PWD}/models:/opt/caffe/models \ 
                      -v ${PWD}/crrn/model:/opt/caffe/crrn/model \
                      tbpp_crnn:gpu bash

For convenince this command is executed when running `./run.bash`.
  

### Models

1. pre-trained model on SynthText (used for training):
[Dropbox](https://www.dropbox.com/s/kpv17f3syio95vn/model_pre_train_syn.caffemodel?dl=0); 
[BaiduYun](https://pan.baidu.com/s/1htV2j4K)

2. model trained on ICDAR 2015 Incidental Text (used for testing):
[Dropbox](https://www.dropbox.com/s/9znpiqpah8rir9c/model_icdar15.caffemodel?dl=0); 
[BaiduYun](https://pan.baidu.com/s/1bqekTun)
    
    Please place the above models in "./models/"
    
    If your data is hugely different from ICDAR 2015 Incidental Text，you'd better train it on your own data based on the pre-trained model on SynthText.

3. CRNN model:
[Dropbox](https://www.dropbox.com/s/kmi62qxm9z08o6h/model_crnn.t7?dl=0);
[BaiduYun](https://pan.baidu.com/s/1jJwmneI)

    Please place the crnn model in "./crnn/model/"


### Demo 

Download the ICDAR 2015 model and place it in "./models/"
  ```Shell
  python examples/text/demo.py
  ```
The detection results and recognition results are in "./demo_images"

### Train

#### Create lmdb data

1. convert ground truth into "xml" form: [example.xml](./data/example.xml)
    
2. create train/test lists (train.txt / test.txt) in "./data/text/" with the following form: 

        path_to_example1.jpg path_to_example1.xml
        path_to_example2.jpg path_to_example2.xml
            
3. Run "./data/text/creat_data.sh"
    
#### Start training
    
    1. modify the lmdb path in modelConfig.py
    2. Run "python examples/text/train.py"
