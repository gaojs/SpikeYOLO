## 注意事项
    1.  源码链接：https://github.com/BICLab/SpikeYOLO
    2.  权重文件：best.pt(官方预训练模型checkpoint(23M, T=1, D=4))
        1）checkpoint(23M, T=1, D=4):https://drive.google.com/file/d/1YQ29eDUfmaze2jl_UREX4Zeb1u8tpHfl/view?usp=sharing
    3.  来自 李国齐 开源链接
        1）如果要设置后处理置信度筛选，ultralytics/models/yolo/detect/predict.py  第5行添加设置


## 报错修改（已修改）
    1.  运行报错  AttributeError: Can't get attribute 'Conv2d_bn' on <module 'ultralytics.nn.modules.yolo_spikformer' from '/data3/zhou.zhou/project_bidl/00-common/1-SpikeYOLO_20260414/ultralytics/nn/modules/yolo_spikformer.py'>
        1）按照开源链接说明，使用该权重需用 yolo_spikformer_bin.py 替换 yolo_spikformer.py


## 适配修改（已修改）
    1.  固定模型输入形状，如下640x640
        1）ultralytics/engine/predictor.py  第148行same_shapes

    2.  只输出推理结果
        1）ultralytics/nn/modules/yolo_spikformer.py   第289行

    3.  不支持算子适配修改 
        1）quant4
            ultralytics/nn/modules/yolo_spikformer.py  第65-70/82-83行

    4.  resize3d算子不支持导致部分编译到cpu
        1）ultralytics/nn/tasks.py                     第86-91行

    5.  降维同等替换
        1）ultralytics/nn/modules/yolo_spikformer.py   第46-69、201-203、218-220行


## 操作步骤
    1.  在x86用python3.8安装环境依赖及lygnor1.22.0及以上
            pip3 install -r requirements.txt

    2.  执行编译 ./test_build.sh    # 在x86上

    3.  推理测试 ./test_demo.sh     # 在盒子或板卡上


## 编译和推理测试
    1）网络推理位置：ultralytics/engine/predictor.py  第136行

    -- 20260416
    1.  用(1,3,640,640)lyngor1.22.0 编译 best.pt 通过，推理速度 1.8fps，误差率约 8%