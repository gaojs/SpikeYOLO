import lyngor as lyn
# import os; os.environ["DMLC_LOG_DEBUG"] = "1"   # 开启编译打印
import zz_conf; zz_conf.is_use_sdk = False        # 编译时不推理

model_file  = "best.pt"
model_save  = "model_spikeyolov8_320_c16"
model_shape = (1,3,320,320)
model_node  = "images"

def lyn_build(model_file, save_path, model_shape, run_batch=1):
    mod = lyn.DLModel()
    mod.load(model_file, model_type="Pytorch", inputs_dict={model_node:model_shape},
                in_type="uint8",                  # 前处理：指定输入数据类型
                out_type="float16",               # 后处理：指定输出数据类型
                variance=(255,),                  # 前处理：添加归一化参数
                out_transpose_axis=[(0,2,1)],     # 后处理：输出转置(1,84,N)->(1,N,84)样式
                transpose_axis=[(0,3,1,2)])       # 前处理：输入转置NCHW->NHWC
    offline_builder = lyn.Builder(target="apu")
    offline_builder.build(mod.graph, mod.param, out_path=save_path, run_batch=run_batch,
                            serialize=False, core_mem_mode=0, post_mode=500, lut_opt=1,
                            num_cores=16)
    print(' ###[lyn_build] model build end! save path is', save_path)

if __name__=="__main__":
    lyn_build(model_file, model_save, model_shape)
