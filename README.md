## Installation

docker pull  junhyung5544/my_openpcdet:spconv_2_1_21_python3_6

make container mounting your workspace

git clone https://github.com/konyul/fusion_openpcdet.git

git clone https://github.com/open-mmlab/mmdetection3d.git in the fusion_openpcdet repository


in fusion_openpcdet repository,

pip install -r requirement.txt

python setup.py develop

in mmdetection3d repository,

pip install -v -e.

in fusion_openpcdet repository

python setup.py develop

pip install mmcv-full==1.4.8 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html

pip install mmdet==2.19.0

pip install mmsegmentation==0.20.0

apt-get install libgl1-mesa-glx

check if pcdet/ops/voxel/voxel_layer is compiled
  if not python setup.py develop on fusion_openpcdet repository







## License

`OpenPCDet` is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement
`OpenPCDet` is an open source project for LiDAR-based 3D scene perception that supports multiple
LiDAR-based perception models as shown above. Some parts of `PCDet` are learned from the official released codes of the above supported methods. 
We would like to thank for their proposed methods and the official implementation.   

We hope that this repo could serve as a strong and flexible codebase to benefit the research community by speeding up the process of reimplementing previous works and/or developing new methods.


## Citation 
If you find this project useful in your research, please consider cite:


```
@misc{openpcdet2020,
    title={OpenPCDet: An Open-source Toolbox for 3D Object Detection from Point Clouds},
    author={OpenPCDet Development Team},
    howpublished = {\url{https://github.com/open-mmlab/OpenPCDet}},
    year={2020}
}
```

## Contribution
Welcome to be a member of the OpenPCDet development team by contributing to this repo, and feel free to contact us for any potential contributions. 

