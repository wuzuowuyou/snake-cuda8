# Deep Snake for Real-Time Instance Segmentation


## from zju3dv-snake
[https://github.com/zju3dv/snake]

## Requirements
* Python 3.7
* torch  1.0.0
* torchvision 0.2.2.post3

## 几处修改说明
* 由于一些原因，项目还是用的cuda8，libtorch1.0的，为了能把snake用在工程中，改成cuda8，pytorch1.0并转libtorch1.0
* dcn用的这个仓库的，[https://github.com/xi11xi19/CenterNet2TorchScript],原版的dcn在该环境下的pytorch也是可以训练测试的，但是转pt不成功，会报错。用CenterNet2TorchScript仓库的dcn是可以转pt并用它写好的c++也是可以在libtorch加载运行。
./lib/csrc/extreme_utils正常编译
./lib/csrc/roi_align_layer不需要编译

* ./lib/config/config.py
```
cfg.save_ep = 1  #一个epoch保存模型############################### 
cfg.eval_ep = 5000000 ###5000000epoch测试，就是让他不要测试#######################
cfg.iteration_save_my = 1500  #迭代多少step保存一个模型
```
会在./myfile/save_model文件夹下生成iteration.pth，由于数据几百万，训练一轮需要几天，一个epoch就需要几天，万一哪次中断训练前面就白训练了，因为没有模型保存。设置iteration_save_my可以根据指定步长保存最新的。

## 训练过程
* 用sbd格式的数据训练。数据准备：在文件夹./data/sbd目录下：
```
├── annotations
│   ├── sbd_train_instance.json
│   ├── sbd_trainval_instance.json
│   └── sbd_val_instance.json
└── img
    ├── 1C4HJXENXKW553851_N_1995_20190101_flg2701_0.jpg
    ├── 1C4HJXENXKW553851_N_1995_20190101_flg2701_-10.jpg
    ├── 1C4HJXENXKW553851_N_1995_20190101_flg2701_10.jpg
    ├── 1C4HJXENXKW553851_N_1995_20190101_flg2701_-13.jpg

测试和验证图片都需要在img文件夹，可以先把训练样本做好。
测试样本再做好，把图片拷贝到img文件夹即可。
```
* ./configs/sbd_snake.yaml改类别数
heads: {'ct_hm': 1, 'wh': 2}  ##ct_hm为类别数，不包含背景类
还有lr，batch_size，num_workers等参数


训练指令：
```
python train_net.py --cfg_file configs/sbd_snake.yaml model sbd_snake
```

显示效果图指令：
```
python run.py --type demo --cfg_file configs/sbd_snake.yaml demo_path demo_images ct_score 0.3
```

