import sys
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                           QComboBox, QProgressBar, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QFont
import cv2
from main import Mode, main

# 设置默认目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
INPUT_DIR = os.path.join(DATA_DIR, 'input')
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
MODELS_DIR = os.path.join(DATA_DIR, 'models')

# 创建必要的目录
for directory in [DATA_DIR, INPUT_DIR, OUTPUT_DIR, MODELS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

class VideoProcessingThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, source_path, target_path, device, mode):
        super().__init__()
        self.source_path = source_path
        self.target_path = target_path
        self.device = device
        self.mode = mode

    def run(self):
        try:
            # 获取视频总帧数
            cap = cv2.VideoCapture(self.source_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            # 修改main函数调用，添加进度回调
            def progress_callback(frame_number):
                progress = int((frame_number / total_frames) * 100)
                self.progress.emit(progress)

            # 调用main函数
            main(self.source_path, self.target_path, self.device, self.mode, progress_callback)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("足球分析系统")
        self.setMinimumSize(800, 600)
        self.setup_ui()

    def setup_ui(self):
        # 创建主窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 标题
        title_label = QLabel("足球分析系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 24, QFont.Bold))
        layout.addWidget(title_label)

        # 输入视频选择
        input_layout = QHBoxLayout()
        self.input_label = QLabel("输入视频：")
        self.input_path_label = QLabel("未选择文件")
        self.input_button = QPushButton("选择文件")
        self.input_button.clicked.connect(self.select_input_video)
        input_layout.addWidget(self.input_label)
        input_layout.addWidget(self.input_path_label)
        input_layout.addWidget(self.input_button)
        layout.addLayout(input_layout)

        # 输出视频选择
        output_layout = QHBoxLayout()
        self.output_label = QLabel("输出视频：")
        self.output_path_label = QLabel("未选择文件")
        self.output_button = QPushButton("选择文件")
        self.output_button.clicked.connect(self.select_output_video)
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.output_path_label)
        output_layout.addWidget(self.output_button)
        layout.addLayout(output_layout)

        # 模式选择
        mode_layout = QHBoxLayout()
        self.mode_label = QLabel("分析模式：")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([mode.value for mode in Mode])
        mode_layout.addWidget(self.mode_label)
        mode_layout.addWidget(self.mode_combo)
        layout.addLayout(mode_layout)

        # 设备选择
        device_layout = QHBoxLayout()
        self.device_label = QLabel("运行设备：")
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "cuda"])
        device_layout.addWidget(self.device_label)
        device_layout.addWidget(self.device_combo)
        layout.addLayout(device_layout)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # 开始按钮
        self.start_button = QPushButton("开始分析")
        self.start_button.clicked.connect(self.start_processing)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                font-size: 16px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        layout.addWidget(self.start_button)

        # 设置样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QLabel {
                font-size: 14px;
            }
            QComboBox {
                padding: 5px;
                border: 1px solid #ccc;
                border-radius: 3px;
            }
            QPushButton {
                padding: 5px 15px;
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)

    def select_input_video(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "选择输入视频", INPUT_DIR, "Video Files (*.mp4 *.avi *.mov)"
        )
        if file_name:
            self.input_path_label.setText(file_name)

    def select_output_video(self):
        file_name, _ = QFileDialog.getSaveFileName(
            self, "选择输出视频", OUTPUT_DIR, "Video Files (*.mp4)"
        )
        if file_name:
            self.output_path_label.setText(file_name)

    def start_processing(self):
        if self.input_path_label.text() == "未选择文件":
            QMessageBox.warning(self, "警告", "请选择输入视频文件")
            return
        if self.output_path_label.text() == "未选择文件":
            QMessageBox.warning(self, "警告", "请选择输出视频文件")
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.start_button.setEnabled(False)
        
        # 创建并启动处理线程
        self.processing_thread = VideoProcessingThread(
            self.input_path_label.text(),
            self.output_path_label.text(),
            self.device_combo.currentText(),
            Mode(self.mode_combo.currentText())
        )
        self.processing_thread.progress.connect(self.update_progress)
        self.processing_thread.finished.connect(self.processing_finished)
        self.processing_thread.error.connect(self.processing_error)
        self.processing_thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def processing_finished(self):
        self.progress_bar.setVisible(False)
        self.start_button.setEnabled(True)
        QMessageBox.information(self, "完成", "视频处理完成！")

    def processing_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.start_button.setEnabled(True)
        QMessageBox.critical(self, "错误", f"处理过程中出现错误：{error_msg}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_()) 