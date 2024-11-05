import os
import sys
current_file_path = os.path.realpath(__file__) #返回当前文件的绝对路径
run_path = os.path.dirname(os.path.dirname(current_file_path)) #获取当前脚本所在的父目录
ultralyticsmain_path = os.path.dirname(run_path)
sys.path.append(ultralyticsmain_path)
from ultralytics import YOLO

from ultralytics import YOLO

if __name__ == '__main__':
    # 加载模型
    model = YOLO('')  #权重文件路径,建议使用绝对路径
    # 验证模型
    metrics=model.val(
        val=False,  # (bool) 在训练期间进行验证/测试
        data='',  # (str) 数据集的路径,建议使用绝对路径
        split='test',  # (str) 用于验证的数据集拆分，例如'val'、'test'或'train'
        batch=6,  # (int) 每批的图像数量（-1 为自动批处理）
        imgsz=640,  # 输入图像的大小，可以是整数或w，h
        device='',  # 运行的设备，例如 cuda device=0 或 device=0,1,2,3 或 device=cpu
        workers=12,  # 数据加载的工作线程数（每个DDP进程）
        save_json=True,  # 保存结果到JSON文件
        save_hybrid=False,  # 保存标签的混合版本（标签 + 额外的预测）
        conf=0.001,  # 检测的目标置信度阈值（默认为0.25用于预测，0.001用于验证）
        iou=0.6,  # 非极大值抑制 (NMS) 的交并比 (IoU) 阈值
        project='',  # 测试结果保存路径（可选）
        name='',  # 实验名称，结果保存在'project/name'目录下（可选）
        max_det=3,  # 每张图像的最大检测数
        half=False,  # 使用半精度 (FP16)
        dnn=False,  # 使用OpenCV DNN进行ONNX推断
        plots=True,  # 在训练/验证期间保存图像
    )

    print(f"mAP50-95: {metrics.box.map}") # map50-95
    print(f"mAP50: {metrics.box.map50}")  # map50
    print(f"mAP75: {metrics.box.map75}")  # map75
    speed_metrics = metrics.speed
    total_time = sum(speed_metrics.values())
    fps = 1000 / total_time
    print(f"FPS: {fps}") # FPS

