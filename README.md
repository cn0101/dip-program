# 足球分析系统

这是一个基于计算机视觉的足球视频分析系统，可以检测和分析足球比赛中的各种元素。

## 功能特点

1. 球场检测：自动识别足球场地的边界和关键点
2. 球员检测：检测场上的球员、守门员和裁判
3. 足球检测：追踪足球的位置和运动轨迹
4. 球员跟踪：跟踪每个球员的移动轨迹

## 系统要求

- Python 3.8+
- CUDA支持（可选，用于GPU加速）
- 足够的系统内存（建议8GB以上）

## 安装步骤

1. 克隆仓库：
```bash
git clone [https://github.com/cn0101/dip-program]
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 下载预训练模型：
将预训练模型文件放置在 `data/models/` 目录下：
- football-player-detection.pt
- football-pitch-detection.pt
- football-ball-detection.pt

## 使用方法

运行以下命令启动图形界面：
```bash
python gui.py
```

## 开源项目引用

本项目使用了以下开源项目：

1. [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
   - 用于目标检测和关键点检测
   - 许可证：AGPL-3.0
   - 版本：8.3.86

2. [Supervision](https://github.com/roboflow/supervision)
   - 用于视频处理和标注
   - 许可证：MIT
   - 版本：0.23.0

3. [PyQt5](https://www.riverbankcomputing.com/software/pyqt/)
   - 用于构建图形用户界面
   - 许可证：GPL v3
   - 版本：5.15.9

4. [OpenCV](https://opencv.org/)
   - 用于图像处理和视频处理
   - 许可证：Apache 2.0
   - 版本：4.11.0

5. [ByteTrack](https://github.com/ifzhang/ByteTrack)
   - 用于多目标跟踪
   - 许可证：MIT
   - 版本：1.0.0

## 注意事项

1. 确保输入视频质量良好，画面稳定
2. 建议使用GPU进行加速处理
3. 处理大视频文件时请确保有足够的磁盘空间

## 许可证

本项目基于以下开源许可证：
- AGPL-3.0 (YOLOv8)
- MIT (Supervision, ByteTrack)
- GPL v3 (PyQt5)
- Apache 2.0 (OpenCV)

## 联系方式

[联系方式]
