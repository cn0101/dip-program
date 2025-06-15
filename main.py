import argparse
from enum import Enum
from typing import Iterator, List

import os
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from sports.common.ball import BallTracker, BallAnnotator
from sports.configs.soccer import SoccerPitchConfiguration

# 设置项目根目录和模型路径
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data', 'models', 'football-player-detection.pt')
PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data', 'models', 'football-pitch-detection.pt')
BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data', 'models', 'football-ball-detection.pt')

# 定义检测类别ID
BALL_CLASS_ID = 0        # 足球
GOALKEEPER_CLASS_ID = 1  # 守门员
PLAYER_CLASS_ID = 2      # 普通球员
REFEREE_CLASS_ID = 3     # 裁判

# 视频处理参数
STRIDE = 60  # 视频帧采样间隔
CONFIG = SoccerPitchConfiguration()  # 足球场配置

# 定义标注颜色
COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']

# 初始化各种标注器
# 顶点标签标注器：用于标注球场关键点
VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
    color=[sv.Color.from_hex(color) for color in CONFIG.colors],
    text_color=sv.Color.from_hex('#FFFFFF'),
    border_radius=5,
    text_thickness=1,
    text_scale=0.5,
    text_padding=5,
)

# 边缘标注器：用于标注球场边界线
EDGE_ANNOTATOR = sv.EdgeAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    thickness=2,
    edges=CONFIG.edges,
)

# 三角形标注器：用于标注球场区域
TRIANGLE_ANNOTATOR = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    base=20,
    height=15,
)

# 边界框标注器：用于标注检测到的目标
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)

# 椭圆标注器：用于标注球员位置
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)

# 边界框标签标注器：用于显示目标类别
BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)

# 椭圆标签标注器：用于显示球员ID
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)

class Mode(Enum):
    """
    系统运行模式枚举类
    """
    PITCH_DETECTION = 'PITCH_DETECTION'    # 球场检测模式
    PLAYER_DETECTION = 'PLAYER_DETECTION'  # 球员检测模式
    BALL_DETECTION = 'BALL_DETECTION'      # 足球检测模式
    PLAYER_TRACKING = 'PLAYER_TRACKING'    # 球员跟踪模式

def run_pitch_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    运行球场检测
    
    参数:
        source_video_path: 输入视频路径
        device: 运行设备（'cpu'或'cuda'）
    
    返回:
        生成器，产生标注后的视频帧
    """
    # 加载球场检测模型
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    
    # 逐帧处理视频
    for frame in frame_generator:
        # 运行检测
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        # 标注检测结果
        annotated_frame = frame.copy()
        annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(
            annotated_frame, keypoints, CONFIG.labels)
        yield annotated_frame

def run_player_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    运行球员检测
    
    参数:
        source_video_path: 输入视频路径
        device: 运行设备（'cpu'或'cuda'）
    
    返回:
        生成器，产生标注后的视频帧
    """
    # 加载球员检测模型
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    
    # 逐帧处理视频
    for frame in frame_generator:
        # 运行检测
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        # 标注检测结果
        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(annotated_frame, detections)
        yield annotated_frame

def run_ball_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    运行足球检测
    
    参数:
        source_video_path: 输入视频路径
        device: 运行设备（'cpu'或'cuda'）
    
    返回:
        生成器，产生标注后的视频帧
    """
    # 加载足球检测模型
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    
    # 初始化足球跟踪器和标注器
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    # 定义检测回调函数
    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    # 初始化切片器，用于处理大尺寸图像
    slicer = sv.InferenceSlicer(
        callback=callback,
        overlap_filter_strategy=sv.OverlapFilter.NONE,
        slice_wh=(640, 640),
    )

    # 逐帧处理视频
    for frame in frame_generator:
        # 运行检测和跟踪
        detections = slicer(frame).with_nms(threshold=0.1)
        detections = ball_tracker.update(detections)
        
        # 标注检测结果
        annotated_frame = frame.copy()
        annotated_frame = ball_annotator.annotate(annotated_frame, detections)
        yield annotated_frame

def run_player_tracking(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    运行球员跟踪
    
    参数:
        source_video_path: 输入视频路径
        device: 运行设备（'cpu'或'cuda'）
    
    返回:
        生成器，产生标注后的视频帧
    """
    # 加载球员检测模型
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    
    # 初始化跟踪器
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    
    # 逐帧处理视频
    for frame in frame_generator:
        # 运行检测和跟踪
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        # 生成跟踪ID标签
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        # 标注检测结果
        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels=labels)
        yield annotated_frame

def main(source_video_path: str, target_video_path: str, device: str, mode: Mode, progress_callback=None) -> None:
    """
    主函数：运行足球视频分析
    
    参数:
        source_video_path: 输入视频路径
        target_video_path: 输出视频路径
        device: 运行设备（'cpu'或'cuda'）
        mode: 运行模式
        progress_callback: 进度回调函数
    """
    # 获取视频信息
    frame_count = 0
    total_frames = int(cv2.VideoCapture(source_video_path).get(cv2.CAP_PROP_FRAME_COUNT))
    video_info = sv.VideoInfo.from_video_path(source_video_path)

    # 根据模式选择处理函数
    if mode == Mode.PITCH_DETECTION:
        frame_iterator = run_pitch_detection(source_video_path, device)
    elif mode == Mode.PLAYER_DETECTION:
        frame_iterator = run_player_detection(source_video_path, device)
    elif mode == Mode.BALL_DETECTION:
        frame_iterator = run_ball_detection(source_video_path, device)
    elif mode == Mode.PLAYER_TRACKING:
        frame_iterator = run_player_tracking(source_video_path, device)
    else:
        raise ValueError(f"未知模式: {mode}")

    # 处理视频并保存结果
    with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink:
        for frame in frame_iterator:
            sink.write_frame(frame)
            frame_count += 1
            if progress_callback:
                progress_callback(frame_count)

if __name__ == '__main__':
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='足球视频分析系统')
    parser.add_argument('--source_video_path', type=str, required=True)
    parser.add_argument('--target_video_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--mode', type=Mode, default=Mode.PLAYER_DETECTION)
    args = parser.parse_args()
    
    # 运行主函数
    main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=args.device,
        mode=args.mode
    ) 