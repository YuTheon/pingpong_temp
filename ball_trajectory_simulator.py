"""乒乓球轨迹模拟器.

该模块用于模拟从LCM接收到的球位置数据，用于验证可视化系统。
"""

import atexit
import csv
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5 import QtCore, QtGui, QtWidgets

from collections import deque

# 导入自定义模块(仿真时去掉了前面的点，正式运行需要补上)
from .chart_renderer import ChartRenderer
from .interpolation import TrajectoryInterpolator
from .landing_analyzer import LandingAnalyzer
from .trajectory_recorder import TrajectoryRecorder

# 添加exlcm模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))
exlcm_path = os.path.join(os.path.dirname(current_dir), "exlcm")
if exlcm_path not in sys.path:
    sys.path.insert(0, exlcm_path)
    print(f"📁 添加exlcm路径: {exlcm_path}")

# LCM相关导入
try:
    # 先尝试导入lcm库
    import lcm
    
    # 再尝试导入自定义类型
    import exlcm
    
    LCM_AVAILABLE = True
    print("✅ LCM库导入成功，实时数据功能可用")
except ImportError as e:
    LCM_AVAILABLE = False
    print(f"⚠️ LCM库导入失败: {e}")
    print("⚠️ 无法接收实时数据，将使用离线模式")
from PyQt5.QtCore import QEasingCurve, QPropertyAnimation, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFormLayout, 
    QLineEdit,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from plot3D_230704 import plot3D
from utils.logger import logger

import math

class LowPassFilter:
    def __init__(self, alpha, init_value=None):
        self.y = init_value
        self.s = None
        self.alpha = alpha

    def set_alpha(self, alpha):
        self.alpha = alpha

    def filter(self, value, timestamp=None, alpha=None):
        if alpha is not None:
            self.alpha = alpha
        if self.s is None:
            self.s = value
        else:
            self.s = self.alpha * value + (1.0 - self.alpha) * self.s
        return self.s

class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """
        min_cutoff: 最小截止频率。值越小，慢速时越平滑（抖动越少），但延迟越高。
        beta: 速度系数。值越大，高速时延迟越低，但高速时的抖动可能会增加。
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_filter = LowPassFilter(self.compute_alpha(min_cutoff))
        self.dx_filter = LowPassFilter(self.compute_alpha(d_cutoff))
        self.last_timestamp = None

    def compute_alpha(self, cutoff, dt=None):
        if dt is None: return 1.0 # 默认不过滤
        te = 1.0 / cutoff
        tau = 1.0 / (2 * math.pi * te)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, timestamp):
        # x 应该是 np.array([x, y, z])
        if self.last_timestamp is None:
            self.last_timestamp = timestamp
            self.x_filter.s = x
            self.dx_filter.s = np.zeros_like(x)
            return x

        dt = timestamp - self.last_timestamp
        
        # 防止重复时间戳或时间倒流
        if dt <= 0:
            return self.x_filter.s

        self.last_timestamp = timestamp

        # 1. 计算速度 (position derivative)
        dx = (x - self.x_filter.s) / dt
        edx = self.dx_filter.filter(dx, timestamp, alpha=self.compute_alpha(self.d_cutoff, dt))

        # 2. 根据速度计算动态截止频率
        # 速度越大，cutoff 越大，过滤越弱，延迟越低
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(edx)
        
        # 3. 过滤位置
        return self.x_filter.filter(x, timestamp, alpha=self.compute_alpha(cutoff, dt))


class FuturisticButton(QPushButton):

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(
            """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(0, 100, 200, 0.8),
                    stop:1 rgba(0, 50, 150, 0.9));
                border: 2px solid rgba(0, 200, 255, 0.6);
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
                color: white;
                text-align: center;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(0, 150, 255, 0.9),
                    stop:1 rgba(0, 100, 200, 1.0));
                border: 2px solid rgba(0, 255, 255, 0.8);
                transform: scale(1.05);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(0, 80, 160, 1.0),
                    stop:1 rgba(0, 40, 120, 1.0));
                border: 2px solid rgba(0, 180, 255, 1.0);
                padding-top: 14px;
                padding-bottom: 10px;
            }
            QPushButton:disabled {
                background: rgba(50, 50, 50, 0.5);
                border: 2px solid rgba(100, 100, 100, 0.3);
                color: rgba(150, 150, 150, 0.7);
            }
        """
        )
        self.setMinimumHeight(50)
        self.setCursor(Qt.PointingHandCursor)


class RecordButton(QPushButton):
    """录制按钮特殊样式"""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(
            """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(0, 100, 200, 0.8),
                    stop:1 rgba(0, 50, 150, 0.9));
                border: 2px solid rgba(0, 200, 255, 0.6);
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
                color: white;
                text-align: center;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(0, 150, 255, 0.9),
                    stop:1 rgba(0, 100, 200, 1.0));
                border: 2px solid rgba(0, 255, 255, 0.8);
                transform: scale(1.05);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(0, 80, 160, 1.0),
                    stop:1 rgba(0, 40, 120, 1.0));
                border: 2px solid rgba(0, 180, 255, 1.0);
                padding-top: 14px;
                padding-bottom: 10px;
            }
            QPushButton:checked {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255, 59, 48, 0.9),
                    stop:1 rgba(200, 40, 30, 1.0));
                border: 2px solid rgba(255, 100, 100, 0.8);
                color: white;
            }
            QPushButton:checked:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255, 80, 70, 1.0),
                    stop:1 rgba(220, 50, 40, 1.0));
                border: 2px solid rgba(255, 120, 120, 1.0);
            }
            QPushButton:checked:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(200, 40, 30, 1.0),
                    stop:1 rgba(180, 30, 20, 1.0));
                border: 2px solid rgba(255, 140, 140, 1.0);
                padding-top: 14px;
                padding-bottom: 10px;
            }
            QPushButton:disabled {
                background: rgba(50, 50, 50, 0.5);
                border: 2px solid rgba(100, 100, 100, 0.3);
                color: rgba(150, 150, 150, 0.7);
            }
        """
        )
        self.setMinimumHeight(50)
        self.setCursor(Qt.PointingHandCursor)
        self.setCheckable(True)


class RealtimeRenderButton(QPushButton):
    """实时渲染按钮特殊样式"""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(
            """
            QPushButton {
                background: transparent;
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 12px;
                font-weight: bold;
                color: white;
                text-align: center;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.5);
            }
            QPushButton:pressed {
                background: rgba(255, 255, 255, 0.2);
                border: 1px solid rgba(255, 255, 255, 0.7);
            }
            QPushButton:checked {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255, 59, 48, 0.9),
                    stop:1 rgba(200, 40, 30, 1.0));
                border: 2px solid rgba(255, 100, 100, 0.8);
                color: white;
            }
            QPushButton:checked:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255, 80, 70, 1.0),
                    stop:1 rgba(220, 50, 40, 1.0));
                border: 2px solid rgba(255, 120, 120, 1.0);
            }
            QPushButton:checked:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(200, 40, 30, 1.0),
                    stop:1 rgba(150, 30, 20, 1.0));
                border: 2px solid rgba(255, 80, 80, 1.0);
            }
            QPushButton:disabled {
                background: rgba(50, 50, 50, 0.5);
                border: 1px solid rgba(100, 100, 100, 0.3);
                color: rgba(150, 150, 150, 0.7);
            }
            """
        )
        self.setMinimumHeight(36)
        self.setCursor(Qt.PointingHandCursor)
        self.setCheckable(True)


class ServerConfigDialog(QDialog):
    """远程服务器配置对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Remote Server Configuration")
        self.setModal(True)

        layout = QFormLayout(self)

        self.host_input = QLineEdit()
        self.port_input = QLineEdit()
        self.port_input.setText("7667")  # 默认端口

        layout.addRow("Server Address:", self.host_input)
        layout.addRow("Port:", self.port_input)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)


class ProgramDiagnosisDialog(QDialog):
    """程序诊断结果对话框"""

    def __init__(self, program_path, parent=None):
        super().__init__(parent)
        self.program_path = program_path
        self.setWindowTitle("Program Diagnosis")
        self.setModal(True)
        self.setFixedSize(700, 500)

        layout = QVBoxLayout(self)

        # 标题
        title_label = QLabel(f"🔍 程序诊断结果: {os.path.basename(program_path)}")
        title_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #FFC107; margin-bottom: 10px;"
        )
        layout.addWidget(title_label)

        # 程序路径
        path_label = QLabel(f"📁 程序路径: {program_path}")
        path_label.setStyleSheet("color: white; margin-bottom: 10px;")
        layout.addWidget(path_label)

        # 诊断结果文本区域
        self.diagnosis_text = QListWidget()
        self.diagnosis_text.setStyleSheet(
            """
            QListWidget {
                background: rgba(0, 0, 0, 0.3);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 4px;
                color: white;
                font-family: 'Courier New', monospace;
                font-size: 11px;
            }
        """
        )
        layout.addWidget(self.diagnosis_text)

        # 按钮
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        refresh_btn = QPushButton("🔄 重新诊断")
        refresh_btn.setStyleSheet(
            """
            QPushButton {
                background: rgba(255, 193, 7, 0.2);
                border: 1px solid rgba(255, 193, 7, 0.5);
                border-radius: 4px;
                padding: 8px 16px;
                color: #FFC107;
                font-size: 12px;
            }
            QPushButton:hover {
                background: rgba(255, 193, 7, 0.3);
                border: 1px solid rgba(255, 193, 7, 0.7);
            }
        """
        )
        refresh_btn.clicked.connect(self.run_diagnosis)

        close_btn = QPushButton("关闭")
        close_btn.setStyleSheet(
            """
            QPushButton {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 4px;
                padding: 8px 20px;
                color: white;
                font-size: 12px;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.2);
                border: 1px solid rgba(255, 255, 255, 0.5);
            }
        """
        )
        close_btn.clicked.connect(self.accept)

        button_layout.addWidget(refresh_btn)
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)

        # 设置对话框样式
        self.setStyleSheet(
            """
            QDialog {
                background: rgba(40, 40, 40, 0.95);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 8px;
            }
        """
        )

        # 运行诊断
        self.run_diagnosis()

    def run_diagnosis(self):
        """运行诊断"""
        self.diagnosis_text.clear()
        self.add_diagnosis_item("🔍 开始诊断程序...", "info")
        
        # 检查文件是否存在
        if not os.path.exists(self.program_path):
            self.add_diagnosis_item("❌ 文件不存在", "error")
            return
        
        # 检查文件权限
        stat_info = os.stat(self.program_path)
        self.add_diagnosis_item(f"📁 文件类型: 权限 {oct(stat_info.st_mode)[-3:]}", "info")
        
        if not os.access(self.program_path, os.X_OK):
            self.add_diagnosis_item("❌ 文件没有执行权限", "error")
            self.add_diagnosis_item("💡 建议运行: chmod +x " + self.program_path, "suggestion")
        else:
            self.add_diagnosis_item("✅ 文件有执行权限", "success")
        
        # 检查文件大小
        file_size = stat_info.st_size
        self.add_diagnosis_item(f"📏 文件大小: {file_size} 字节", "info")
        
        # 检查文件类型
        try:
            result = subprocess.run(['file', self.program_path], capture_output=True, text=True)
            if result.returncode == 0:
                file_type = result.stdout.strip()
                self.add_diagnosis_item(f"📋 文件类型: {file_type}", "info")
            else:
                self.add_diagnosis_item("⚠️ 无法确定文件类型", "warning")
        except Exception as e:
            self.add_diagnosis_item(f"⚠️ 文件类型检查失败: {e}", "warning")
        
        # 检查程序依赖
        self.add_diagnosis_item("🔍 检查程序依赖...", "info")
        try:
            result = subprocess.run(['ldd', self.program_path], capture_output=True, text=True)
            if result.returncode == 0:
                missing_libs = []
                found_libs = []
                for line in result.stdout.split('\n'):
                    if '=>' in line:
                        if 'not found' in line:
                            missing_libs.append(line.strip())
                        else:
                            found_libs.append(line.strip())
                
                self.add_diagnosis_item(f"✅ 找到的库: {len(found_libs)} 个", "success")
                
                if missing_libs:
                    self.add_diagnosis_item(f"❌ 缺失的库: {len(missing_libs)} 个", "error")
                    for lib in missing_libs[:5]:  # 只显示前5个
                        self.add_diagnosis_item(f"   {lib}", "error")
                    if len(missing_libs) > 5:
                        self.add_diagnosis_item(f"   ... 还有 {len(missing_libs) - 5} 个缺失库", "error")
                else:
                    self.add_diagnosis_item("✅ 所有依赖库都已找到", "success")
            else:
                self.add_diagnosis_item(f"⚠️ 无法检查依赖: {result.stderr}", "warning")
        except Exception as e:
            self.add_diagnosis_item(f"⚠️ 依赖检查失败: {e}", "warning")
        
        # 检查工作目录
        working_dir = os.path.dirname(os.path.abspath(self.program_path))
        self.add_diagnosis_item(f"📁 工作目录: {working_dir}", "info")
        
        if os.path.exists(working_dir):
            self.add_diagnosis_item("✅ 工作目录存在", "success")
            try:
                files = os.listdir(working_dir)
                self.add_diagnosis_item(f"📋 工作目录内容: {len(files)} 个文件/目录", "info")
            except Exception as e:
                self.add_diagnosis_item(f"⚠️ 无法列出工作目录内容: {e}", "warning")
        else:
            self.add_diagnosis_item("❌ 工作目录不存在", "error")
        
        # 尝试测试运行程序
        self.add_diagnosis_item("🧪 测试运行程序...", "info")
        try:
            # 使用timeout防止程序卡死
            result = subprocess.run(
                [self.program_path, '--help'],  # 尝试显示帮助信息
                capture_output=True,
                text=True,
                timeout=5,
                cwd=working_dir
            )
            self.add_diagnosis_item(f"✅ 程序可以启动，退出码: {result.returncode}", "success")
            if result.stdout:
                self.add_diagnosis_item(f"📤 标准输出: {result.stdout[:100]}...", "info")
            if result.stderr:
                self.add_diagnosis_item(f"📤 错误输出: {result.stderr[:100]}...", "info")
        except subprocess.TimeoutExpired:
            self.add_diagnosis_item("⚠️ 程序启动超时（可能正在运行）", "warning")
        except Exception as e:
            self.add_diagnosis_item(f"❌ 程序启动失败: {e}", "error")
        
        self.add_diagnosis_item("🔍 诊断完成", "info")

    def add_diagnosis_item(self, text, level="info"):
        """添加诊断项目到列表"""
        item = QListWidgetItem(text)
        
        # 根据级别设置颜色
        if level == "error":
            item.setForeground(QtGui.QColor("#FF6B6B"))  # 红色
        elif level == "warning":
            item.setForeground(QtGui.QColor("#FFD93D"))  # 黄色
        elif level == "success":
            item.setForeground(QtGui.QColor("#6BCF7F"))  # 绿色
        elif level == "suggestion":
            item.setForeground(QtGui.QColor("#4ECDC4"))  # 青色
        else:  # info
            item.setForeground(QtGui.QColor("#FFFFFF"))  # 白色
        
        self.diagnosis_text.addItem(item)
        # 自动滚动到底部
        self.diagnosis_text.scrollToBottom()


class SettingsDialog(QDialog):
    """设置对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setFixedSize(500, 200)

        layout = QVBoxLayout(self)

        # 标题
        title_label = QLabel("Data Collection Program Settings")
        title_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: white; margin-bottom: 10px;"
        )
        layout.addWidget(title_label)

        # 采集程序路径设置
        form_layout = QFormLayout()

        # 路径输入框
        self.program_path_input = QLineEdit()
        self.program_path_input.setPlaceholderText(
            "Enter the absolute path to the data collection program"
        )
        self.program_path_input.setStyleSheet(
            """
            QLineEdit {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 4px;
                padding: 8px;
                color: white;
                font-size: 12px;
            }
            QLineEdit:focus {
                border: 1px solid rgba(255, 255, 255, 0.6);
            }
        """
        )

        # 浏览按钮
        browse_btn = QPushButton("Browse")
        browse_btn.setStyleSheet(
            """
            QPushButton {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 4px;
                padding: 8px 16px;
                color: white;
                font-size: 12px;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.2);
                border: 1px solid rgba(255, 255, 255, 0.5);
            }
            QPushButton:pressed {
                background: rgba(255, 255, 255, 0.3);
                border: 1px solid rgba(255, 255, 255, 0.7);
            }
        """
        )
        browse_btn.clicked.connect(self.browse_program)

        # 路径输入行布局
        path_layout = QHBoxLayout()
        path_layout.addWidget(self.program_path_input)
        path_layout.addWidget(browse_btn)

        form_layout.addRow("Collection Program Path:", path_layout)
        layout.addLayout(form_layout)

        # 说明文字
        info_label = QLabel(
            "This program will be executed when clicking 'Local Monitor' button"
        )
        info_label.setStyleSheet(
            "color: rgba(255, 255, 255, 0.7); font-size: 11px; margin: 10px 0;"
        )
        layout.addWidget(info_label)

        # 诊断按钮
        diagnose_btn = QPushButton("🔍 Diagnose Program")
        diagnose_btn.setStyleSheet(
            """
            QPushButton {
                background: rgba(255, 193, 7, 0.2);
                border: 1px solid rgba(255, 193, 7, 0.5);
                border-radius: 4px;
                padding: 8px 16px;
                color: #FFC107;
                font-size: 12px;
                margin: 5px 0;
            }
            QPushButton:hover {
                background: rgba(255, 193, 7, 0.3);
                border: 1px solid rgba(255, 193, 7, 0.7);
            }
            QPushButton:pressed {
                background: rgba(255, 193, 7, 0.4);
                border: 1px solid rgba(255, 193, 7, 0.9);
            }
        """
        )
        diagnose_btn.clicked.connect(self.diagnose_program)
        layout.addWidget(diagnose_btn)

        # 按钮
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")

        for btn in [ok_btn, cancel_btn]:
            btn.setStyleSheet(
                """
                QPushButton {
                    background: rgba(255, 255, 255, 0.1);
                    border: 1px solid rgba(255, 255, 255, 0.3);
                    border-radius: 4px;
                    padding: 8px 20px;
                    color: white;
                    font-size: 12px;
                    min-width: 60px;
                }
                QPushButton:hover {
                    background: rgba(255, 255, 255, 0.2);
                    border: 1px solid rgba(255, 255, 255, 0.5);
                }
                QPushButton:pressed {
                    background: rgba(255, 255, 255, 0.3);
                    border: 1px solid rgba(255, 255, 255, 0.7);
                }
            """
            )

        ok_btn.clicked.connect(self.save_and_accept)
        cancel_btn.clicked.connect(self.reject)

        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

        # 设置对话框样式
        self.setStyleSheet(
            """
            QDialog {
                background: rgba(40, 40, 40, 0.95);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 8px;
            }
        """
        )

        # 加载保存的路径
        self.load_saved_path()

    def browse_program(self):
        """浏览选择程序文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Data Collection Program",
            "",
            "All Files (*);;Python Files (*.py);;Executable Files (*.exe *.app)",
        )
        if file_path:
            self.program_path_input.setText(file_path)

    def diagnose_program(self):
        """诊断程序启动问题"""
        program_path = self.program_path_input.text().strip()
        if not program_path:
            QMessageBox.warning(
                self,
                "No Program Path",
                "Please enter a program path first.",
            )
            return
        
        # 创建诊断结果对话框
        dialog = ProgramDiagnosisDialog(program_path, self)
        dialog.exec_()

    def load_saved_path(self):
        """加载保存的程序路径"""
        try:
            config_file = os.path.join(os.path.dirname(__file__), "settings.conf")
            if os.path.exists(config_file):
                with open(config_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("collection_program="):
                            path = line.split("=", 1)[1].strip()
                            self.program_path_input.setText(path)
                            break
        except Exception as e:
            print(f"⚠️ 加载设置失败: {e}")

    def save_path(self):
        """保存程序路径"""
        try:
            config_file = os.path.join(os.path.dirname(__file__), "settings.conf")
            with open(config_file, "w", encoding="utf-8") as f:
                f.write(f"collection_program={self.program_path_input.text()}\n")
            print(f"✅ 设置已保存: {config_file}")
        except Exception as e:
            print(f"❌ 保存设置失败: {e}")

    def get_program_path(self):
        """获取程序路径"""
        return self.program_path_input.text().strip()
        
    def save_and_accept(self):
        """保存设置并关闭对话框"""
        self.save_path()
        self.accept()

class MainWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.menu_btn = None
        self.record_btn = None
        self.realtime_render_btn = None
        self.reset_charts_btn = None
        self.button_frame = None
        self.speed_label = None

    def resizeEvent(self, event):
        super().resizeEvent(event)

        # 通知父类（BallTrajectorySimulator）更新UI位置
        if hasattr(self, "simulator_instance") and self.simulator_instance:
            # 延迟调用，确保窗口大小调整完成
            QTimer.singleShot(50, self.simulator_instance._update_ui_positions)
        else:
            # 如果没有父类引用，使用默认位置（仅作为备用）
            if self.menu_btn:
                self.menu_btn.move(30, self.height() - 70)
            if self.record_btn:
                self.record_btn.move(30, 130)
            if self.button_frame:
                self.button_frame.move(
                    self.width() - self.button_frame.width() - 30,
                    self.height() - self.button_frame.height() - 30,
                )
            if self.speed_label:
                self.speed_label.move(self.width() - self.speed_label.width() - 30, 30)
            if hasattr(self, "speed_chart_label"):
                self.speed_chart_label.move(
                    self.width() - self.speed_chart_label.width() - 30, 80
                )

from .trajectory_processor import TrajectoryProcessor  # 导入新拆分的处理器


class BallTrajectorySimulator:
    """乒乓球轨迹模拟器类.

    负责从CSV文件读取球位置数据，并通过LCM模拟发送位置消息。
    """

    def __init__(self, save_folder_path=None, on_close_callback=None):
        """初始化模拟器."""
        self.csv_file_path = None
        self.positions = []  # 存储位置数据
        self.timestamps = []  # 存储时间戳数据（秒）
        self.current_index = 0
        self.is_paused = False
        self.data_source = None  # 当前数据源类型
        self.server_config = None  # 远程服务器配置

        # --- 新增：去噪和平滑缓冲区 ---
        # self.raw_data_buffer = deque() # 存储 (timestamp, x, y, z)
        # self.buffer_duration = 0.1     # 0.1秒延迟
        self.last_valid_pos = None     # 上一个确认有效的坐标，用于距离过滤
        self.max_jump_distance = 300.0 # 最大允许跳变距离(mm)，超过此值视为误检
        # ---------
        # --- 新增：One-Euro Filter ---
        # min_cutoff=1.0: 慢速时非常平滑
        # beta=0.007: 高速时快速响应 (需要根据实际数据单位微调)
        # 如果你的单位是 mm，速度可能达到几千，beta 需要设得很小，例如 0.001 或 0.0001
        # 如果单位是 m，速度是 10 左右，beta 可以设为 0.5 或 1.0
        # 鉴于你的数据 X,Y,Z 看起来是毫米 (如 559.071)，beta 建议设小一点。
        self.one_euro_filter = OneEuroFilter(min_cutoff=1.5, beta=0.05, d_cutoff=1.0)
        
        self.last_valid_pos = None     
        self.max_jump_distance = 400.0 # 稍微放宽一点，避免高速球被误删
        # ----------------------------
 

        # 存档文件夹路径
        self.save_folder_path = save_folder_path
        print(
            f"🎮 模拟器初始化，存档文件夹: {save_folder_path if save_folder_path else '全局目录'}"
        )

        self.processor = TrajectoryProcessor(save_folder_path)


        # 关闭回调函数
        self.on_close_callback = on_close_callback

        # LCM线程安全相关变量
        self.lcm_lock = threading.Lock()  # LCM操作线程锁
        self.lcm_operation_in_progress = False  # LCM操作进行中标志

        # 重置所有状态变量，确保重新打开时状态正确
        self.reset_simulator_state()

        # 动态播放列表（包含原始数据点和插值点）
        self.playback_timestamps = []  # 播放时间戳列表
        self.playback_positions = []  # 播放位置列表
        self.playback_index = 0  # 播放索引

        # 视频录制相关变量
        self.is_recording = False
        self.record_fps = 30  # 录制帧率
        self.video_writer = None
        self.record_timer = None

        # --- 新增：发球评估模式 --------
        self.is_evaluating_serve = False
        self.serve_data = [] # 存储发球过程的轨迹

        # 初始化各个模块
        self.interpolator = TrajectoryInterpolator()
        self.landing_analyzer = LandingAnalyzer(save_folder_path)
        # 使用模拟器格式记录轨迹数据，便于直接重放
        self.trajectory_recorder = TrajectoryRecorder(save_folder_path, use_simulator_format=True)
        self.chart_renderer = ChartRenderer(save_folder_path)

        # 轨迹相关变量
        self.complete_trajectory = []  # 完整的轨迹队列（包含原始数据和插值数据）
        self.trajectory_index = 0  # 当前轨迹索引
        self.is_rendering = False  # 渲染状态标志

        # LCM相关变量
        self.lcm_instance = None
        self.lcm_subscription = None
        self.lcm_thread = None
        self.lcm_running = False
        self.real_time_positions = []  # 实时接收的位置数据
        self.real_time_timestamps = []  # 实时接收的时间戳数据

        # 加载球台角点
        try:
            # 使用相对路径：从当前文件位置向上两级目录，然后到Yolov5-table-edge
            corners_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "Yolov5-table-edge",
                "pts_3d.npy",
            )
            corners = np.load(corners_file)
        except:
            # 如果文件不存在，使用默认值（以球台中心为原点）
            corners = np.array(
                [
                    [-1370, -762.5, 0],
                    [1370, -762.5, 0],
                    [1370, 762.5, 0],
                    [-1370, 762.5, 0],
                ]
            )
            logger.warning("使用默认球台角点坐标（以球台中心为原点）")

        window_size = (1200, 800)
        self.plt = plot3D(window_size, corners, None, None, True, 5)

        # 定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_position)

        # 创建主窗口
        self.main_widget = MainWidget()
        # 设置对模拟器的引用，用于resizeEvent
        self.main_widget.simulator_instance = self
        self.layout = QVBoxLayout(self.main_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.plt.view)
        self.main_widget.setLayout(self.layout)
        self.main_widget.setWindowTitle("Ping Pong Ball Trajectory Simulator")

        # 设置窗口图标（应用程序图标已在main函数中设置）
        try:
            from PyQt5.QtWidgets import QApplication

            app = QApplication.instance()
            if app and not app.windowIcon().isNull():
                self.main_widget.setWindowIcon(app.windowIcon())
                print("✅ 窗口图标已从应用程序图标设置")
        except Exception as e:
            print(f"⚠️ 设置窗口图标失败: {e}")

        self.main_widget.resize(1200, 900)

        # 创建功能按钮（直接显示二级菜单选项）
        # 启动跟踪按钮
        self.local_monitor_btn = QPushButton("StartTacker", self.main_widget)
        self.local_monitor_btn.setStyleSheet(
            """
            QPushButton {
                background: transparent;
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 14px;
                color: white;
                font-weight: 500;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.5);
            }
            QPushButton:pressed {
                background: rgba(255, 255, 255, 0.2);
                border: 1px solid rgba(255, 255, 255, 0.7);
                padding-top: 9px;
                padding-bottom: 7px;
            }
        """
        )
        self.local_monitor_btn.setFixedSize(150, 36)
        self.local_monitor_btn.move(30, 30)
        self.local_monitor_btn.clicked.connect(self.start_local_monitor)
        self.local_monitor_btn.show()
        self.main_widget.local_monitor_btn = self.local_monitor_btn

        # 本地轨迹按钮
        self.local_trajectory_btn = QPushButton("Local Trajectory", self.main_widget)
        self.local_trajectory_btn.setStyleSheet(
            """
            QPushButton {
                background: transparent;
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 14px;
                color: white;
                font-weight: 500;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.5);
            }
            QPushButton:pressed {
                background: rgba(255, 255, 255, 0.2);
                border: 1px solid rgba(255, 255, 255, 0.7);
                padding-top: 9px;
                padding-bottom: 7px;
            }
        """
        )
        self.local_trajectory_btn.setFixedSize(150, 36)
        self.local_trajectory_btn.move(30, 80)
        self.local_trajectory_btn.clicked.connect(self.start_local_trajectory)
        self.local_trajectory_btn.show()
        self.main_widget.local_trajectory_btn = self.local_trajectory_btn

        # 创建录制按钮
        self.record_btn = QPushButton("Record", self.main_widget)
        self.record_btn.setStyleSheet(
            """
            QPushButton {
                background: transparent;
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 14px;
                color: white;
                font-weight: 500;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.5);
            }
            QPushButton:pressed {
                background: rgba(255, 255, 255, 0.2);
                border: 1px solid rgba(255, 255, 255, 0.7);
                padding-top: 9px;
                padding-bottom: 7px;
            }
            QPushButton:checked {
                background: rgba(255, 59, 48, 0.8);
                color: white;
                border: 1px solid rgba(255, 59, 48, 0.6);
            }
            QPushButton:checked:hover {
                background: rgba(255, 59, 48, 0.9);
                border: 1px solid rgba(255, 59, 48, 0.7);
            }
            QPushButton:checked:pressed {
                background: rgba(255, 59, 48, 0.7);
                border: 1px solid rgba(255, 59, 48, 0.8);
                padding-top: 9px;
                padding-bottom: 7px;
            }
        """
        )
        self.record_btn.setFixedSize(150, 36)
        self.record_btn.setCheckable(True)
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.show()
        self.main_widget.record_btn = self.record_btn

        # 创建实时渲染按钮
        self.realtime_render_btn = RealtimeRenderButton("Real-time Render", self.main_widget)
        self.realtime_render_btn.setFixedSize(150, 36)
        self.realtime_render_btn.clicked.connect(self.toggle_realtime_render)
        self.realtime_render_btn.show()
        self.main_widget.realtime_render_btn = self.realtime_render_btn

        # 创建重置按钮
        self.reset_charts_btn = QPushButton("Reset Charts", self.main_widget)
        self.reset_charts_btn.setStyleSheet(
            """
            QPushButton {
                background: transparent;
                border: 1px solid rgba(255, 193, 7, 0.5);
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 12px;
                font-weight: bold;
                color: rgba(255, 193, 7, 0.9);
                text-align: center;
            }
            QPushButton:hover {
                background: rgba(255, 193, 7, 0.1);
                border: 1px solid rgba(255, 193, 7, 0.8);
                color: rgb(255, 193, 7);
            }
            QPushButton:pressed {
                background: rgba(255, 193, 7, 0.2);
                border: 1px solid rgba(255, 193, 7, 1.0);
                color: white;
            }
            """
        )
        self.reset_charts_btn.setFixedSize(150, 36)
        self.reset_charts_btn.clicked.connect(self.reset_chart_data)
        self.reset_charts_btn.show()
        self.main_widget.reset_charts_btn = self.reset_charts_btn


        # [新增] 创建发球评估按钮
        self.eval_serve_btn = QPushButton("Evaluate Serve", self.main_widget)
        self.eval_serve_btn.setStyleSheet(
            """
            QPushButton {
                background: transparent;
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 14px;
                color: white;
                font-weight: 500;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.5);
            }
            QPushButton:checked {
                background: rgba(230, 126, 34, 0.8); /* 橙色背景表示激活 */
                border: 1px solid rgba(230, 126, 34, 1.0);
            }
            """
        )
        self.eval_serve_btn.setFixedSize(150, 36)
        self.eval_serve_btn.setCheckable(True) # 设置为可选中状态
        self.eval_serve_btn.clicked.connect(self.toggle_serve_evaluation)
        self.eval_serve_btn.show()
        self.main_widget.eval_serve_btn = self.eval_serve_btn # 保存引用

        # 在 eval_serve_btn 下方添加一个查看分布的按钮
        self.view_stats_btn = QPushButton("Serve History", self.main_widget)
        self.view_stats_btn.setStyleSheet(self.local_monitor_btn.styleSheet()) # 复用样式
        self.view_stats_btn.setFixedSize(150, 36)
        self.view_stats_btn.clicked.connect(self.show_serve_history_stats)
        self.view_stats_btn.show()
        # 更新 UI 位置逻辑中也要加上这一行

        # 创建控制按钮层（初始隐藏）
        self.button_frame = QFrame(self.main_widget)
        self.button_frame.setAttribute(Qt.WA_TranslucentBackground)
        self.button_frame.setStyleSheet("background: rgba(0,0,0,0);")
        self.button_frame.setFrameShape(QFrame.NoFrame)
        self.button_layout = QHBoxLayout(self.button_frame)
        self.button_layout.setContentsMargins(0, 0, 0, 0)
        self.button_layout.setSpacing(10)

        # 创建控制按钮
        self.start_btn = QPushButton("Start")
        self.pause_btn = QPushButton("Pause")
        self.reset_btn = QPushButton("Reset")
        self.switch_source_btn = QPushButton("Switch Source")

        # 设置控制按钮样式
        control_button_style = """
            QPushButton {
                background: transparent;
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 14px;
                color: white;
                font-weight: 500;
                min-width: 80px;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.5);
            }
            QPushButton:pressed {
                background: rgba(255, 255, 255, 0.2);
                border: 1px solid rgba(255, 255, 255, 0.7);
                padding-top: 9px;
                padding-bottom: 7px;
            }
        """

        for btn in [
            self.start_btn,
            self.pause_btn,
            self.reset_btn,
            self.switch_source_btn,
        ]:
            btn.setStyleSheet(control_button_style)
            self.button_layout.addWidget(btn)

        self.button_frame.setLayout(self.button_layout)
        self.button_frame.resize(360, 50)  # 恢复原来的宽度
        self.button_frame.move(
            self.main_widget.width() - self.button_frame.width() - 30,
            self.main_widget.height() - self.button_frame.height() - 30,
        )
        self.button_frame.hide()  # 初始隐藏控制按钮
        self.main_widget.button_frame = self.button_frame

        # 连接按钮信号
        self.start_btn.clicked.connect(self.start)
        self.pause_btn.clicked.connect(self.pause)
        self.reset_btn.clicked.connect(self.reset_all_data)
        # self.switch_source_btn.clicked.connect(self.show_server_config)

        # 添加球速显示标签
        self.speed_label = QLabel(self.main_widget)
        self.speed_label.setText("time: 00:00:00\nSpeed: 0.0 m/s\nShots: 0")
        self.speed_label.setStyleSheet(
            """
            QLabel {
                background: transparent;
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 28px;
                color: white;
                font-weight: 500;
                line-height: 1.2;
            }
        """
        )
        self.speed_label.setFixedSize(400, 200)  # 增加宽度从300到400，确保完整显示内容
        
        # 调试信息：显示标签尺寸和内容
        print(f"📏 速度标签尺寸: {self.speed_label.width()}x{self.speed_label.height()}")
        print(f"📝 速度标签内容: {self.speed_label.text()}")
        
        # 初始位置将在_update_ui_positions中设置
        self.speed_label.raise_()
        self.speed_label.show()
        self.main_widget.speed_label = self.speed_label

        # 添加速度折线图显示区域
        self.speed_chart_label = QLabel(self.main_widget)
        self.speed_chart_label.setStyleSheet(
            """
            QLabel {
                background: transparent;
                border: none;
                padding: 5px;
            }
            """
        )
        self.speed_chart_label.setFixedSize(450, 300)
        # 调整位置，避免与变宽的速度标签重叠
        self.speed_chart_label.move(
            self.main_widget.width() - self.speed_chart_label.width() - 50, 80
        )
        self.speed_chart_label.raise_()
        self.speed_chart_label.show()
        self.main_widget.speed_chart_label = self.speed_chart_label

        # 替换原有热力图UI初始化部分：
        self.heatmap_canvas = QLabel("No landing data", self.main_widget)
        self.heatmap_canvas.setAlignment(Qt.AlignRight)
        self.heatmap_canvas.setStyleSheet(
            "color: white; font-size: 12px; background: transparent;"
        )
        self.heatmap_canvas.setFixedSize(280, 300)  # 扩大热力图尺寸
        # 初始位置将在_update_ui_positions中设置
        self.heatmap_canvas.show()
        self.main_widget.heatmap_canvas = self.heatmap_canvas

        # 添加散点图显示区域
        self.scatter_canvas = QLabel("No landing data", self.main_widget)
        self.scatter_canvas.setAlignment(Qt.AlignCenter)
        self.scatter_canvas.setStyleSheet(
            "color: white; font-size: 12px; background: transparent;"
        )
        self.scatter_canvas.setFixedSize(140, 300)  # 与热力图相同尺寸
        # 初始位置将在_update_ui_positions中设置
        self.scatter_canvas.show()
        self.main_widget.scatter_canvas = self.scatter_canvas

        # 初始化时显示空的热力图和散点图
        self.update_heatmap_display()
        self.update_scatter_display()

        # 初始化时显示速度折线图
        self.update_speed_chart()

        # 初始化完成后更新UI位置
        QTimer.singleShot(100, self._update_ui_positions)

        # 强制刷新布局
        QTimer.singleShot(200, self._force_refresh_layout)

        # 初始化训练时长相关变量
        self.training_start_time = time.time()  # 程序启动时间作为训练开始时间
        self.total_training_time = 0  # 总训练时长（秒）
        self.last_save_time = time.time()  # 上次保存时间
        
        # 加载累积的训练时长
        self.load_accumulated_training_time()
        
        # 验证并重置异常的训练时长
        if self.validate_and_reset_training_time():
            print("🔄 训练时长已重置为正常值")

        # 初始化训练时长更新定时器（每秒更新一次）
        self.training_timer = QTimer()
        self.training_timer.timeout.connect(self.update_training_time_display)
        self.training_timer.start(1000)  # 每秒更新一次

        # 立即显示初始训练时长
        QTimer.singleShot(100, self.update_training_time_display)

        # 设置关闭事件处理，返回到主界面而不是退出程序
        self.main_widget.closeEvent = self.handle_close_event

        # 注册信号处理器，确保程序退出时正确清理
        import signal
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # 注册退出时清理函数
        atexit.register(self.cleanup)

        # 初始化实时渲染按钮状态
        self._init_realtime_render_button_state()

        self.main_widget.show()

    def show_serve_history_stats(self):
        """展示发球落点分布统计图"""
        history_file = os.path.join(self.save_folder_path or ".", "serve_stats/serve_history.csv")
        if not os.path.exists(history_file):
            QMessageBox.information(self.main_widget, "空空如也", "还没有任何发球历史数据。")
            return
            
        # 这里你可以复用 ChartRenderer 的逻辑
        # 或者直接弹出一个基于你现有 heatmap 逻辑生成的汇总图
        QMessageBox.information(self.main_widget, "统计提示", "当前历史落点已同步到下方的 Heatmap 和 Scatter 图中。")
        self.update_heatmap_display()

    def save_serve_to_history(self, report):
            """将单次发球结果存入历史数据库 (CSV)"""
            import csv
            # 确定存档路径
            history_dir = os.path.join(self.save_folder_path or ".", "serve_stats")
            os.makedirs(history_dir, exist_ok=True)
            history_file = os.path.join(history_dir, "serve_history.csv")
            
            file_exists = os.path.exists(history_file)
            
            with open(history_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Time", "Max_Speed_ms", "Peak_H_mm", "Landing_X", "Landing_Y", "Duration_s"])
                
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    f"{report['max_speed']:.2f}",
                    f"{report['peak_height']:.1f}",
                    f"{report['landing_x']:.1f}",
                    f"{report['landing_y']:.1f}",
                    f"{report['duration']:.2f}"
                ])

    def _smooth_and_filter(self):
        """
        从缓冲区中提取、过滤并平滑数据
        返回: (timestamp, [x, y, z]) 或者 None
        """
        current_time = time.time()
        
        # 1. 如果缓冲区为空，返回None
        if not self.raw_data_buffer:
            return None

        # 2. 检查缓冲区最老的数据是否已经"成熟" (达到0.1s延迟)
        # 实际上我们看缓冲区长度即可，或者看时间戳差值
        oldest_ts, _, _, _ = self.raw_data_buffer[0]
        newest_ts, _, _, _ = self.raw_data_buffer[-1]
        
        if (newest_ts - oldest_ts) < self.buffer_duration:
            # 缓冲区数据还不够长（还没积累够0.1s的数据），暂时不渲染
            return None

        # 3. 弹出最老的一个点进行处理
        ts, raw_x, raw_y, raw_z = self.raw_data_buffer.popleft()
        current_pos = np.array([raw_x, raw_y, raw_z])

        # --- 步骤A: 误检过滤 (基于距离) ---
        if self.last_valid_pos is not None:
            # 计算与上一个有效点的欧氏距离
            dist = np.linalg.norm(current_pos - self.last_valid_pos)
            
            # 如果距离大得离谱 (例如0.01秒内飞了30厘米以上)，视为误检
            # 注意：如果是新回合的发球，距离也会很大，需要额外逻辑重置，
            # 但通常发球前会有停顿，这里简单处理：如果距离太大，丢弃。
            # 为了防止连续丢弃导致无法追踪新球，可以加个计数器，
            # 如果连续丢弃超过N次，就强制接受新位置（视为瞬移或新回合）。
            if dist > self.max_jump_distance:
                print(f"🗑️ 剔除噪点: 距离 {dist:.1f} > {self.max_jump_distance}")
                return None # 丢弃该点，不更新UI

        # --- 步骤B: 平滑 (简单移动平均) ---
        # 利用缓冲区里剩下的点(也就是未来的点)和当前点做平均
        # 取缓冲区前3-5个点做平均
        avg_x, avg_y, avg_z = raw_x, raw_y, raw_z
        count = 1
        
        # 向前看(缓冲区内的点就是"未来"的点)
        look_ahead = min(len(self.raw_data_buffer), 4) 
        for i in range(look_ahead):
            _, bx, by, bz = self.raw_data_buffer[i]
            # 简单的距离检查，防止把远处的噪点也平均进去了
            if np.linalg.norm(np.array([bx,by,bz]) - current_pos) < self.max_jump_distance:
                avg_x += bx
                avg_y += by
                avg_z += bz
                count += 1
        
        final_pos = np.array([avg_x/count, avg_y/count, avg_z/count])
        
        # 更新上一个有效点
        self.last_valid_pos = final_pos
        return ts, final_pos

    def _signal_handler(self, signum, frame):
        """处理系统信号，确保程序正确退出"""
        try:
            print(f"🔄 收到系统信号 {signum}，开始清理...")
            
            # 保存当前训练时长
            try:
                total_seconds = self.calculate_training_time()
                self.save_training_time_to_archive(total_seconds)
                print(f"💾 信号处理时训练时长已保存: {total_seconds:.0f}秒")
            except Exception as e:
                print(f"⚠️ 信号处理时保存训练时长失败: {e}")
            
            # 执行安全关闭
            self.safe_shutdown()
            
            # 额外的进程清理保险措施
            self._cleanup_all_trajectory_simulators()
            
        except Exception as e:
            print(f"❌ 信号处理失败: {e}")
            # 即使出错也要清理进程
            try:
                self._cleanup_all_trajectory_simulators()
            except:
                pass
        finally:
            # 强制退出程序
            import sys
            sys.exit(0)

    def reset_simulator_state(self):
        """重置模拟器状态，确保重新打开时状态正确"""
        # 重置播放状态
        self.is_paused = False
        self.is_rendering = False
        self.trajectory_index = 0
        self.current_index = 0

        # 重置轨迹数据
        self.complete_trajectory = []
        self.playback_timestamps = []
        self.playback_positions = []
        self.playback_index = 0

        # 重置实时数据
        self.real_time_positions = []
        self.real_time_timestamps = []
        self._realtime_trajectory_index = 0

        # 重置LCM状态
        self.lcm_running = False

        # 重置录制状态
        self.is_recording = False

        print("🔄 模拟器状态已重置")

    def update_speed_display(self, speed, shot_count):
        """更新球速显示，包括训练时长、球速和拍数"""
        try:
            # 获取格式化的训练时长
            training_time_str = self.get_formatted_training_time()
            
            # 简化的数值验证
            if not isinstance(speed, (int, float)) or np.isnan(speed) or np.isinf(speed):
                speed = 0.0
                
            if not isinstance(shot_count, int) or shot_count < 0:
                shot_count = 0

            # 更新标签文本
            self.speed_label.setText(
                f"time: {training_time_str}\nSpeed: {speed:.1f} m/s\nShots: {shot_count}"
            )
            
        except Exception as e:
            print(f"❌ 更新球速显示失败: {e}")
            # 使用默认值
            self.speed_label.setText("time: 00:00:00\nSpeed: 0.0 m/s\nShots: 0")

    def calculate_training_time(self):
        """计算训练时长（从程序启动到现在的总时长）"""
        try:
            if hasattr(self, 'training_start_time') and self.training_start_time:
                # 计算当前训练时长
                current_session_time = time.time() - self.training_start_time
                # 总训练时长 = 累积时长 + 当前会话时长
                total_time = self.total_training_time + current_session_time
                return total_time
            else:
                return self.total_training_time
        except Exception as e:
            print(f"❌ 计算训练时长失败: {e}")
            return self.total_training_time

    def start_training_timer(self):
        """开始训练计时"""
        if self.training_start_time is None:
            self.training_start_time = time.time()
            print("⏱️ 训练计时开始")

    def pause_training_timer(self):
        """暂停训练计时"""
        if self.training_start_time is not None:
            # 累加到总训练时长
            current_time = time.time()
            self.total_training_time += current_time - self.training_start_time
            self.training_start_time = None

            # 保存累计训练时长到存档文件
            self.save_training_time()

            print(f"⏸️ 训练计时暂停，累计时长: {self.total_training_time:.1f}秒")

    def save_training_time(self):
        """保存累计训练时长到存档文件"""
        try:
            if self.save_folder_path:
                # 确保目录存在
                os.makedirs(self.save_folder_path, exist_ok=True)

                # 保存累计训练时长（秒）
                training_file = os.path.join(self.save_folder_path, "training_time.txt")
                with open(training_file, "w", encoding="utf-8") as f:
                    f.write(str(int(self.total_training_time)))

                print(f"💾 训练时长已保存到存档: {self.total_training_time:.1f}秒")
        except Exception as e:
            print(f"❌ 保存训练时长失败: {e}")

    def reset_training_timer(self):
        """重置训练计时"""
        self.training_start_time = None
        self.total_training_time = 0

        # 保存重置后的训练时长到存档文件
        self.save_training_time()

        print("🔄 训练计时已重置")

    def update_training_time_display(self):
        """更新训练时长显示（每秒更新一次）"""
        try:
            # 计算当前训练时长
            total_seconds = self.calculate_training_time()
            
            # 转换为时:分:秒格式
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = int(total_seconds % 60)
            training_time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            # 更新速度标签显示
            if hasattr(self, 'speed_label') and self.speed_label:
                current_text = self.speed_label.text()
                lines = current_text.split('\n')
                if len(lines) >= 3:
                    # 更新第一行（训练时长）
                    lines[0] = f"time: {training_time_str}"
                    
                    # 重新组合文本
                    new_text = '\n'.join(lines)
                    self.speed_label.setText(new_text)
            
            # 每60秒自动保存一次训练时长到存档
            current_time = time.time()
            if current_time - self.last_save_time >= 60:  # 60秒保存一次
                self.save_training_time_to_archive(total_seconds)
                self.last_save_time = current_time
                
        except Exception as e:
            print(f"❌ 更新训练时长显示失败: {e}")
            logger.error(f"更新训练时长显示失败: {str(e)}")

    def handle_close_event(self, event):
        """处理关闭事件，返回到主界面而不是退出程序"""
        try:
            print("🔄 模拟器关闭，正在清理资源...")

            # 停止渲染
            self.is_rendering = False
            print("⏹️ 渲染已停止")

            # 保存当前训练时长
            if (
                hasattr(self, "training_start_time")
                and self.training_start_time is not None
            ):
                self.pause_training_timer()
                print("⏹️ 训练时长已保存")

            # 停止所有定时器
            if hasattr(self, "timer") and self.timer.isActive():
                self.timer.stop()
                print("⏹️ 主定时器已停止")

            if hasattr(self, "training_timer") and self.training_timer.isActive():
                self.training_timer.stop()
                print("⏹️ 训练定时器已停止")

            # 清理资源
            self.cleanup()

            # 隐藏模拟器窗口
            self.main_widget.hide()
            print("✅ 模拟器已隐藏，资源已清理")

            # 调用关闭回调函数，通知主菜单显示
            if self.on_close_callback:
                self.on_close_callback()

            # 接受关闭事件，但不退出程序
            event.accept()

        except Exception as e:
            print(f"❌ 处理关闭事件失败: {e}")
            event.accept()

    def start_local_monitor(self):
        """启动本地监视模式"""
        try:
            # 获取采集程序路径
            program_path = self.get_collection_program_path()
            if not program_path:
                QMessageBox.warning(
                    self.main_widget,
                    "No Program Path",
                    "Please set the data collection program path in Settings first.",
                )
                return

            if not os.path.exists(program_path):
                QMessageBox.critical(
                    self.main_widget,
                    "Program Not Found",
                    f"The specified program does not exist:\n{program_path}\n\nPlease check the path in Settings.",
                )
                return

            # 检查文件权限
            if not os.access(program_path, os.X_OK):
                print(f"⚠️ 文件没有执行权限: {program_path}")
                print("🔧 尝试添加执行权限...")
                try:
                    os.chmod(program_path, 0o755)
                    print("✅ 已添加执行权限")
                except Exception as e:
                    print(f"❌ 无法添加执行权限: {e}")
                    QMessageBox.critical(
                        self.main_widget,
                        "Permission Error",
                        f"The program file does not have execute permission:\n{program_path}\n\nPlease run: chmod +x {program_path}",
                    )
                    return

            # 启动采集程序
            print(f"🚀 启动采集程序: {program_path}")
            
            # 获取程序所在目录作为工作目录
            working_dir = os.path.dirname(os.path.abspath(program_path))
            print(f"📁 工作目录: {working_dir}")

            # 使用subprocess启动程序，添加更多启动参数
            try:
                # 设置环境变量，确保C++程序能找到必要的库
                env = os.environ.copy()
                
                # 添加常见的库路径
                if 'LD_LIBRARY_PATH' in env:
                    env['LD_LIBRARY_PATH'] = f"{working_dir}:{env['LD_LIBRARY_PATH']}"
                else:
                    env['LD_LIBRARY_PATH'] = working_dir
                
                # 设置当前工作目录
                env['PWD'] = working_dir
                
                print(f"🔧 环境变量设置:")
                print(f"   LD_LIBRARY_PATH: {env['LD_LIBRARY_PATH']}")
                print(f"   PWD: {env['PWD']}")

                # 尝试多种启动方式
                self.collection_process = None
                startup_success = False
                
                # 方式1: 直接启动（推荐）
                try:
                    print("🔄 尝试方式1: 直接启动...")
                    # 为轨迹模拟器添加适当的参数
                    simulator_args = [
                        program_path,
                        "-i", "10-50",  # 设置较快的发送间隔 (10-50ms)
                        "-v",           # 启用详细输出模式
                        "-l", "-1",      # 无限循环
                        "-d"            # 显示实时数据终端窗口
                    ]
                    
                    self.collection_process = subprocess.Popen(
                        #simulator_args,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        text=True,
                        cwd=working_dir,
                        env=env,
                        preexec_fn=os.setsid if hasattr(os, 'setsid') else None,  # 创建新进程组
                    )
                    # 等待一小段时间检查进程是否启动成功
                    time.sleep(0.5)
                    if self.collection_process.poll() is None:  # 进程仍在运行
                        startup_success = True
                        print("✅ 方式1成功: 直接启动")
                        
                        # 启动一个线程来监控采集程序的输出
                        import threading
                        def monitor_collection_output():
                            try:
                                # 读取采集程序的输出以便调试
                                while self.collection_process and self.collection_process.poll() is None:
                                    output = self.collection_process.stdout.readline()
                                    if output:
                                        print(f"📡 采集程序输出: {output.strip()}")
                                    time.sleep(0.1)
                            except Exception as e:
                                print(f"🔍 采集程序输出监控结束: {e}")
                        
                        monitor_thread = threading.Thread(target=monitor_collection_output, daemon=True)
                        monitor_thread.start()
                        
                    else:
                        # 进程已退出，获取错误信息
                        stdout, stderr = self.collection_process.communicate()
                        print(f"⚠️ 方式1失败，进程退出:")
                        print(f"   退出码: {self.collection_process.returncode}")
                        if stderr:
                            print(f"   错误输出: {stderr}")
                        if stdout:
                            print(f"   标准输出: {stdout}")
                        
                except Exception as e:
                    print(f"❌ 方式1失败: {e}")
                
                # 方式2: 通过shell启动（如果方式1失败）
                if not startup_success:
                    try:
                        print("🔄 尝试方式2: 通过shell启动...")
                        # 构建带参数的shell命令
                        shell_cmd = f"cd '{working_dir}' && '{program_path}' -i 10-50 -v -l -1"
                        
                        self.collection_process = subprocess.Popen(
                            shell_cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            shell=True,
                            env=env,
                            preexec_fn=os.setsid if hasattr(os, 'setsid') else None,
                        )
                        
                        time.sleep(0.5)
                        if self.collection_process.poll() is None:
                            startup_success = True
                            print("✅ 方式2成功: 通过shell启动")
                        else:
                            stdout, stderr = self.collection_process.communicate()
                            print(f"⚠️ 方式2失败，进程退出:")
                            print(f"   退出码: {self.collection_process.returncode}")
                            if stderr:
                                print(f"   错误输出: {stderr}")
                            if stdout:
                                print(f"   标准输出: {stdout}")
                                
                    except Exception as e:
                        print(f"❌ 方式2失败: {e}")
                
                # 方式3: 使用绝对路径并添加调试信息
                if not startup_success:
                    try:
                        print("🔄 尝试方式3: 使用绝对路径启动...")
                        # 检查程序依赖
                        print(f"🔍 检查程序依赖...")
                        try:
                            result = subprocess.run(['ldd', program_path], capture_output=True, text=True)
                            if result.returncode == 0:
                                print("📋 程序依赖库:")
                                for line in result.stdout.split('\n'):
                                    if '=>' in line and 'not found' not in line:
                                        print(f"   {line.strip()}")
                            else:
                                print(f"⚠️ 无法检查依赖: {result.stderr}")
                        except Exception as e:
                            print(f"⚠️ 依赖检查失败: {e}")
                        
                        # 尝试使用完整路径启动（带参数）
                        abs_args = [
                            os.path.abspath(program_path),
                            "-i", "10-50",  # 设置较快的发送间隔
                            "-v",           # 启用详细输出模式
                            "-l", "-1"      # 无限循环
                        ]
                        
                        self.collection_process = subprocess.Popen(
                            abs_args,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            cwd=working_dir,
                            env=env,
                        )
                        
                        time.sleep(0.5)
                        if self.collection_process.poll() is None:
                            startup_success = True
                            print("✅ 方式3成功: 使用绝对路径启动")
                        else:
                            stdout, stderr = self.collection_process.communicate()
                            print(f"⚠️ 方式3失败，进程退出:")
                            print(f"   退出码: {self.collection_process.returncode}")
                            if stderr:
                                print(f"   错误输出: {stderr}")
                            if stdout:
                                print(f"   标准输出: {stdout}")
                                
                    except Exception as e:
                        print(f"❌ 方式3失败: {e}")
                
                # 检查启动结果
                if not startup_success or not self.collection_process:
                    error_msg = "所有启动方式都失败了，请检查程序配置和依赖"
                    print(f"❌ {error_msg}")
                    logger.error(error_msg)
                    QMessageBox.critical(
                        self.main_widget,
                        "Start Failed",
                        f"Failed to start the collection program:\n\n{error_msg}\n\n"
                        f"Program: {program_path}\n"
                        f"Working Directory: {working_dir}\n\n"
                        f"Please check:\n"
                        f"1. File permissions (chmod +x)\n"
                        f"2. Required libraries (ldd {program_path})\n"
                        f"3. Program dependencies\n"
                        f"4. Working directory contents"
                    )
                    return

                # 启动成功
                self.data_source = "local_monitor"
                self.button_frame.hide()

                print(f"✅ 采集程序已启动，PID: {self.collection_process.pid}")
                logger.info(f"启动本地监视模式，程序: {program_path}")

                # 停止加载轨迹渲染（如果正在运行）
                if self.is_rendering:
                    self.pause()
                    print("⏸️ 已停止加载轨迹渲染，准备切换到实时渲染模式")

                # 启动LCM订阅
                if LCM_AVAILABLE:
                    self.start_lcm_subscription()
                else:
                    print("⚠️ LCM库不可用，无法接收实时数据")
                    logger.info(f"LCM库不可用，无法接收实时数据")

            except Exception as e:
                error_msg = f"Failed to start collection program: {str(e)}"
                print(f"❌ {error_msg}")
                logger.error(error_msg)
                QMessageBox.critical(
                    self.main_widget,
                    "Start Failed",
                    f"Failed to start the collection program:\n{str(e)}",
                )

        except Exception as e:
            print(f"❌ 启动本地监视模式失败: {str(e)}")
            logger.error(f"启动本地监视模式失败: {str(e)}")

    def get_collection_program_path(self):
        """获取采集程序路径"""
        try:
            config_file = os.path.join(os.path.dirname(__file__), "settings.conf")
            if os.path.exists(config_file):
                with open(config_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("collection_program="):
                            return line.split("=", 1)[1].strip()
            return ""
        except Exception as e:
            print(f"⚠️ 读取程序路径失败: {e}")
            return ""

    def diagnose_program_startup(self, program_path):
        """诊断程序启动问题"""
        print(f"🔍 诊断程序启动问题: {program_path}")
        
        # 检查文件基本信息
        if not os.path.exists(program_path):
            print("❌ 文件不存在")
            return False
            
        # 检查文件类型
        try:
            import magic
            file_type = magic.from_file(program_path)
            print(f"📁 文件类型: {file_type}")
        except ImportError:
            # 如果没有magic库，使用file命令
            try:
                result = subprocess.run(['file', program_path], capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"📁 文件类型: {result.stdout.strip()}")
                else:
                    print(f"⚠️ 无法确定文件类型: {result.stderr}")
            except Exception as e:
                print(f"⚠️ 文件类型检查失败: {e}")
        
        # 检查文件权限
        stat_info = os.stat(program_path)
        print(f"🔐 文件权限: {oct(stat_info.st_mode)[-3:]}")
        print(f"👤 所有者: {stat_info.st_uid}")
        print(f"👥 组: {stat_info.st_gid}")
        
        # 检查文件大小
        file_size = stat_info.st_size
        print(f"📏 文件大小: {file_size} 字节")
        
        # 检查是否为可执行文件
        if not os.access(program_path, os.X_OK):
            print("❌ 文件没有执行权限")
            return False
        
        # 检查程序依赖
        print("🔍 检查程序依赖...")
        try:
            result = subprocess.run(['ldd', program_path], capture_output=True, text=True)
            if result.returncode == 0:
                missing_libs = []
                found_libs = []
                for line in result.stdout.split('\n'):
                    if '=>' in line:
                        if 'not found' in line:
                            missing_libs.append(line.strip())
                        else:
                            found_libs.append(line.strip())
                
                print(f"✅ 找到的库 ({len(found_libs)}):")
                for lib in found_libs[:5]:  # 只显示前5个
                    print(f"   {lib}")
                if len(found_libs) > 5:
                    print(f"   ... 还有 {len(found_libs) - 5} 个库")
                
                if missing_libs:
                    print(f"❌ 缺失的库 ({len(missing_libs)}):")
                    for lib in missing_libs:
                        print(f"   {lib}")
                    return False
                else:
                    print("✅ 所有依赖库都已找到")
            else:
                print(f"⚠️ 无法检查依赖: {result.stderr}")
        except Exception as e:
            print(f"⚠️ 依赖检查失败: {e}")
        
        # 检查工作目录
        working_dir = os.path.dirname(os.path.abspath(program_path))
        print(f"📁 工作目录: {working_dir}")
        
        if os.path.exists(working_dir):
            print(f"✅ 工作目录存在")
            # 列出目录内容
            try:
                files = os.listdir(working_dir)
                print(f"📋 工作目录内容 ({len(files)} 个文件/目录):")
                for file in files[:10]:  # 只显示前10个
                    file_path = os.path.join(working_dir, file)
                    if os.path.isfile(file_path):
                        print(f"   📄 {file}")
                    else:
                        print(f"   📁 {file}")
                if len(files) > 10:
                    print(f"   ... 还有 {len(files) - 10} 个文件/目录")
            except Exception as e:
                print(f"⚠️ 无法列出工作目录内容: {e}")
        else:
            print(f"❌ 工作目录不存在")
            return False
        
        # 尝试测试运行程序
        print("🧪 测试运行程序...")
        try:
            # 使用timeout防止程序卡死
            result = subprocess.run(
                [program_path, '--help'],  # 尝试显示帮助信息
                capture_output=True,
                text=True,
                timeout=5,
                cwd=working_dir
            )
            print(f"✅ 程序可以启动，退出码: {result.returncode}")
            if result.stdout:
                print(f"📤 标准输出: {result.stdout[:200]}...")
            if result.stderr:
                print(f"📤 错误输出: {result.stderr[:200]}...")
            return True
        except subprocess.TimeoutExpired:
            print("⚠️ 程序启动超时（可能正在运行）")
            return True
        except Exception as e:
            print(f"❌ 程序启动失败: {e}")
            return False

    def start_local_trajectory(self):
        """启动本地轨迹模式"""
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_widget,
            "Select Trajectory File",
            "",  # 起始目录
            "CSV Files (*.csv);;All Files (*)",  # 文件过滤器
        )

        if file_path:  # 如果用户选择了文件
            try:
                # 停止实时渲染（如果正在运行）
                if self.lcm_running:
                    self.stop_lcm_subscription()
                    print("⏸️ 已停止实时渲染，准备切换到加载轨迹渲染模式")
                
                # 更新实时渲染按钮状态
                if hasattr(self, 'realtime_render_btn'):
                    self.realtime_render_btn.setChecked(False)
                    self.realtime_render_btn.setText("Real-time Render")
                    print("🔄 实时渲染按钮状态已重置")

                self.csv_file_path = file_path
                self.data_source = "local_trajectory"
                self.button_frame.show()
                self.switch_source_btn.hide()

                # 切换到加载轨迹渲染模式
                self.switch_to_trajectory_mode()

                self.load_positions()

                # 添加详细的数据统计信息
                print(f"🎯 数据加载完成统计:")
                print(f"   📊 原始CSV数据点数: {len(self.positions)}")
                print(f"   📊 时间戳数量: {len(self.timestamps)}")
                print(
                    f"   📊 时间范围: {self.timestamps[0]:.6f}s - {self.timestamps[-1]:.6f}s"
                )
                print(f"   📊 总时长: {self.timestamps[-1] - self.timestamps[0]:.3f}秒")

                logger.info(f"加载本地轨迹文件: {file_path}")
            except Exception as e:
                QMessageBox.critical(
                    self.main_widget, "Error", f"Failed to load file: {str(e)}"
                )
                logger.error(f"加载文件失败: {str(e)}")
        else:
            logger.info("用户取消选择文件")

    # def start_remote_live(self):
    #     """启动远程实况模式"""
    #     self.show_server_config()

    def show_server_config(self):
        """显示服务器配置对话框"""
        dialog = ServerConfigDialog(self.main_widget)
        if dialog.exec_() == QDialog.Accepted:
            self.server_config = {
                "host": dialog.host_input.text(),
                "port": int(dialog.port_input.text()),
            }
            self.data_source = "remote_live"
            self.button_frame.show()
            self.switch_source_btn.show()
            # TODO: 实现远程数据加载
            logger.info(f"连接到远程服务器: {self.server_config}")

    def start(self):
        print(f"🔍 start方法被调用")
        print(f"   self.positions: {len(self.positions) if self.positions else 'None'}")
        print(
            f"   self.timestamps: {len(self.timestamps) if self.timestamps else 'None'}"
        )
        print(f"   self.current_original_index: {self.current_original_index}")

        if not self.positions or not self.timestamps:
            print("❌ 未加载轨迹数据，无法启动播放")
            logger.info("未加载轨迹数据，无法启动播放")
            return

        # 检查是否已经在播放中（非暂停状态且索引>0）
        if not self.is_paused and self.current_original_index > 0:
            print("ℹ️ 已经在播放中，忽略重复点击")
            return

        print(f"✅ 数据已加载，开始播放")
        print(f"   数据点数: {len(self.positions)}")
        print(f"   时间范围: {self.timestamps[0]:.6f}s - {self.timestamps[-1]:.6f}s")

        if self.is_paused:
            self.is_paused = False
        else:
            # 重置播放索引
            self.trajectory_index = 0

            # 从现有数据文件中读取累积的板数和回合数
            # 使用轨迹记录模块重置数据
            self.trajectory_recorder.reset_accumulated_data()

        # 尝试从现有速度数据文件中读取最后的板数和回合数
        try:
            data_dir = "speed_data"
            filepath = os.path.join(data_dir, "speed_data.csv")
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    last_row = None
                    for row in reader:
                        last_row = row

                    if (
                        last_row
                        and "shot_count" in last_row
                        and "rally_count" in last_row
                    ):
                        shot_count = int(last_row["shot_count"])
                        rally_count = int(last_row["rally_count"])
                        # 使用轨迹记录模块设置数据
                        self.trajectory_recorder.shot_count = shot_count
                        self.trajectory_recorder.rally_count = rally_count
                        print(
                            f"📊 从现有数据恢复：板数 {shot_count}, 回合数 {rally_count}"
                        )
        except Exception as e:
            print(f"⚠️ 读取累积数据失败: {e}")
            # 如果读取失败，保持默认值（从0开始）

        # 生成完整的轨迹队列
        self._generate_complete_trajectory()

        # 开始渲染
        self.is_rendering = True

        # 启动训练计时器
        self.start_training_timer()

        self.update_position()

    def pause(self):
        self.is_paused = True
        self.is_rendering = False

        # 暂停训练计时器
        self.pause_training_timer()

    def refresh(self):
        self.pause()
        self.trajectory_index = 0
        if hasattr(self, "start_time"):
            delattr(self, "start_time")
        self.plt.pos_list = [
            np.full([3], None) for _ in range(self.plt.pos_list_memory_lenth)
        ]
        self.plt.n = 0
        self.plt.updatePlot()
        self.load_positions()
        self.start()  # 刷新后自动播放（保持累积的板数和回合数）

    def reset_accumulated_data(self):
        """重置累积的板数和回合数"""
        # 使用轨迹记录模块重置数据
        self.trajectory_recorder.reset_accumulated_data()

        # 更新显示
        shot_count = self.trajectory_recorder.get_shot_count()
        self.update_speed_display(0.0, shot_count)

        # 更新热力图和散点图显示
        self.update_heatmap_display()
        self.update_scatter_display()

        print("🔄 累积数据已重置：板数归零")

    def reset_all_data(self):
        """重置所有数据：清理落地数据和球速数据"""
        try:
            # 暂停播放
            self.pause()

            # 重置播放状态
            self.trajectory_index = 0
            self.complete_trajectory = []

            # 重置累积数据
            self.reset_accumulated_data()

            # 重置训练计时器
            self.reset_training_timer()

            # 清理落地数据文件
            if self.save_folder_path:
                landing_file = os.path.join(
                    self.save_folder_path, "landing_data", "landing_data.csv"
                )
            else:
                landing_file = os.path.join("landing_data", "landing_data.csv")

            if os.path.exists(landing_file):
                # 备份原文件（可选）
                backup_file = landing_file.replace(
                    ".csv", f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                )
                os.rename(landing_file, backup_file)
                print(f"📁 落地数据已备份到: {backup_file}")

            # 清理球速数据文件
            if self.save_folder_path:
                speed_file = os.path.join(
                    self.save_folder_path, "speed_data", "speed_data.csv"
                )
            else:
                speed_file = os.path.join("speed_data", "speed_data.csv")

            if os.path.exists(speed_file):
                # 备份原文件（可选）
                backup_file = speed_file.replace(
                    ".csv", f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                )
                os.rename(speed_file, backup_file)
                print(f"📁 球速数据已备份到: {backup_file}")

            # 重新初始化数据记录
            self.landing_analyzer.init_landing_data_recording()
            self.trajectory_recorder.init_speed_data_recording()

            # 重置3D可视化
            self.plt.pos_list = [
                np.full([3], None) for _ in range(self.plt.pos_list_memory_lenth)
            ]
            self.plt.n = 0
            self.plt.updatePlot()

            # 更新热力图、散点图和速度图表显示
            self.update_heatmap_display()
            self.update_scatter_display()
            self.update_speed_chart()

            print("🔄 所有数据已重置：落地数据、球速数据、板数已清理")

            # 显示确认消息
            QMessageBox.information(
                self.main_widget,
                "Reset Complete",
                "All data has been reset:\n• Landing data cleared\n• Speed data cleared\n• Shot count reset to zero\n\nOriginal data backed up to backup files.",
            )

        except Exception as e:
            print(f"❌ 重置数据失败: {e}")
            QMessageBox.critical(
                self.main_widget,
                "Reset Failed",
                f"Error occurred while resetting data:\n{str(e)}",
            )

    def load_positions(self):
        """从CSV文件加载球位置数据."""
        print(f"🚀 开始加载数据文件: {self.csv_file_path}")
        try:
            all_positions = []
            all_timestamps = []

            with open(self.csv_file_path, "r") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        # 去除行首尾空格和换行符
                        line = line.strip()
                        if not line:  # 跳过空行
                            continue

                        # 尝试检测数据格式并分割
                        parts = None

                        # 首先尝试逗号分隔（CSV格式）
                        if "," in line:
                            parts = line.split(",")
                        # 如果没有逗号，尝试空格分隔
                        elif " " in line:
                            parts = line.split()
                        else:
                            print(f"⚠️ 第{line_num}行数据格式无法识别: {line}")
                            continue

                        if len(parts) >= 4:  # 确保有足够的数据
                            # 第一个值是时间戳，后面三个是X, Y, Z坐标
                            timestamp = float(parts[0])

                            # 检测时间戳单位和坐标单位
                            # 如果时间戳大于1000000000，认为是微秒单位
                            is_microsecond = timestamp > 1000000000

                            if is_microsecond:
                                # 微秒时间戳：保持微秒精度，不转换为秒
                                # timestamp 保持原值（微秒）
                                x = float(parts[1])  # 已经是毫米
                                y = float(parts[2])  # 已经是毫米
                                z = float(parts[3])  # 已经是毫米
                            else:
                                # 秒时间戳：转换为微秒以保持一致性，坐标转换为毫米
                                timestamp = timestamp * 1000000  # 秒转微秒
                                x = float(parts[1]) * 1000  # 转换为mm
                                y = float(parts[2]) * 1000  # 转换为mm
                                z = float(parts[3]) * 1000  # 转换为mm

                            all_timestamps.append(timestamp)
                            all_positions.append([x, y, z])
                        else:
                            print(f"⚠️ 第{line_num}行数据格式不正确: {line}")

                    except (ValueError, IndexError) as e:
                        print(f"⚠️ 第{line_num}行数据解析失败: {line}, 错误: {e}")
                        continue

            print(f"📊 原始CSV数据: {len(all_positions)} 个数据点")

            # 直接使用原始数据，不进行插值处理
            self.timestamps = all_timestamps
            self.positions = all_positions

            # 初始化播放相关变量
            self.playback_index = 0
            self.current_original_index = 0
            self.interpolation_points = []  # 存储当前区间的插值点
            self.interpolation_index = 0  # 插值点索引

            print(f"🎯 数据加载完成:")
            print(f"   📊 原始CSV数据点数: {len(self.positions)}")
            print(f"   📊 时间戳数量: {len(self.timestamps)}")
            print(
                f"   📊 时间范围: {self.timestamps[0]:.6f}s - {self.timestamps[-1]:.6f}s"
            )
            print(f"   📊 总时长: {self.timestamps[-1] - self.timestamps[0]:.3f}秒")

            logger.info(
                f"Loaded {len(self.positions)} ball positions from {self.csv_file_path}"
            )
            logger.info(
                f"Time range: {self.timestamps[0]:.6f}s - {self.timestamps[-1]:.6f}s"
            )
        except Exception as e:
            print(f"❌ 数据加载失败: {str(e)}")
            logger.error(f"Failed to load ball positions: {str(e)}")
            raise

    def update_position(self):

        if self.is_paused or not self.is_rendering:
            return

        print(f"🔄 update_position被调用，轨迹索引: {self.trajectory_index}")

        # 检查是否播放完成
        if self.trajectory_index >= len(self.complete_trajectory):
            print(
                f"✅ 播放完成，轨迹索引: {self.trajectory_index} >= {len(self.complete_trajectory)}"
            )
            # 播放完成，重置索引并停止
            self.trajectory_index = 0
            self.is_rendering = False
            print("🔄 播放完成，重置到开始位置")
            return

        # 获取当前位置
        current_data = self.complete_trajectory[self.trajectory_index]
        pos = current_data["position"]
        current_time = current_data["time"]
        print(f"📍 轨迹点位置: {pos}")

        # 处理球的位置更新（提取的核心逻辑）
        self._process_ball_position_update(
            pos, current_time, self.trajectory_index, is_realtime=False
        )

        # 移动到下一个轨迹点
        self.trajectory_index += 1

        # 计算下一帧的时间间隔
        if self.trajectory_index < len(self.complete_trajectory):
            next_data = self.complete_trajectory[self.trajectory_index]
            next_time = next_data["time"]
            time_interval = next_time - current_time

            delay_ms = int(time_interval * 1000)  # 转换为毫秒
            print(f"⏱️ 下一帧延迟: {delay_ms}ms")

            # 动态调度下一帧
            if self.is_rendering:
                QTimer.singleShot(delay_ms, self.update_position)
        else:
            print("✅ 播放完成，所有轨迹点已处理完毕")
            # 播放完成，重置到开始位置
            self.trajectory_index = 0
            self.is_rendering = False
            print("🔄 重置到开始位置，准备重新播放")

    def _generate_complete_trajectory(self):
        """生成完整的轨迹队列（包含原始数据和插值数据）"""
        # 使用插值模块生成完整轨迹
        self.complete_trajectory = self.interpolator.generate_complete_trajectory(
            self.positions, self.timestamps
        )

        # 使用落点分析模块分析落点
        landing_points = self.landing_analyzer.analyze_landing_from_csv_data(
            self.positions, self.timestamps
        )

        return landing_points

    def _process_ball_position_update(
        self, pos, current_time, trajectory_index, is_realtime=False
    ):
        """处理球的位置更新（提取的核心逻辑，可在实时渲染时复用）

        Args:
            pos: 当前位置坐标 [x, y, z]
            current_time: 当前时间戳
            trajectory_index: 轨迹索引
            is_realtime: 是否为实时数据
        """
        try:
            # 1. 记录轨迹数据到trajectory_data文件
            self.record_trajectory_data_point(pos)

            # 2. 更新3D可视化
            try:
                if hasattr(self, "plt") and self.plt:
                    # 检查pos_list是否有效
                    if hasattr(self.plt, 'pos_list') and self.plt.pos_list:
                        self.plt.addNewBall(pos)
                        self.plt.updatePlot()
                    else:
                        print("⚠️ 3D视图pos_list无效，跳过更新")
            except Exception as e:
                print(f"⚠️ 3D视图更新失败: {e}")
                logger.error(f"3D视图更新失败: {str(e)}")
                # 不中断其他处理流程

            # 3. 计算球速并检测Y轴趋势变化（实时模式下由process_realtime_position_update处理）
            if not is_realtime and trajectory_index > 0:
                # 轨迹数据：从complete_trajectory获取前一个位置
                # 检查索引是否有效
                if (trajectory_index - 1 < len(self.complete_trajectory) and 
                    len(self.complete_trajectory) > 0):
                    prev_data = self.complete_trajectory[trajectory_index - 1]
                    prev_pos = prev_data["position"]
                    prev_time = prev_data["time"]
                    
                    # 使用轨迹记录模块分析球速和趋势
                    speed, y_trend_changed, current_y_trend = (
                        self.trajectory_recorder.analyze_speed_and_trend(
                            pos, prev_pos, current_time, prev_time
                        )
                    )

                    # 更新球速显示
                    shot_count = self.trajectory_recorder.get_shot_count()
                    self.update_speed_display(speed, shot_count)

                    # 当Y轴趋势改变时记录球速数据
                    if y_trend_changed:
                        self.trajectory_recorder.record_speed_data(
                            current_time, speed, pos, prev_pos
                        )
                        # 更新速度折线图（实时模式和轨迹模式都适用）
                        self.update_speed_chart()

            # 4. 更新帧计数器
            self.landing_analyzer.increment_frame_count()
            self.trajectory_recorder.increment_frame_count()

            # 5. 落点分析：当Z<80时触发，检测z轴运动方向转变作为落点
            if pos[2] is not None and pos[2] < 80:
                if is_realtime:
                    # 实时数据：使用专门的实时落点分析方法
                    self._analyze_realtime_landing(pos, current_time)
                else:
                    # 轨迹数据：使用轨迹落点分析方法
                    self._analyze_landing_from_trajectory(trajectory_index)

            # print(f"✅ 球位置更新处理完成: 位置={pos}, 时间={current_time:.3f}s")

        except Exception as e:
            print(f"❌ 处理球位置更新失败: {e}")
            logger.error(f"Failed to process ball position update: {str(e)}")



    def process_realtime_position_update(self, pos, current_time):
        """处理实时位置更新（移除滤波后的高性能版）"""
        try:
            # 直接将输入坐标转为数组，不经过滤波器处理
            raw_pos = np.array([pos[0], pos[1], pos[2]])

            # 1. 极简异常值剔除：仅过滤掉物理上不可能的瞬移点
            if self.last_valid_pos is not None:
                dist = np.linalg.norm(raw_pos - self.last_valid_pos)
                # 如果 10ms 内球移动超过 50cm，视为无效噪点，直接丢弃
                if dist > 500.0: 
                    return 

            # 更新有效点记录
            self.last_valid_pos = raw_pos
            self.current_time = current_time
            self.frame_count = getattr(self, 'frame_count', 0) + 1

            # 2. 计算速度与趋势分析
            if hasattr(self, "prev_realtime_pos") and self.prev_realtime_pos is not None:
                # 记录速度
                speed, y_trend_changed, current_y_trend = (
                    self.trajectory_recorder.analyze_speed_and_trend(
                        raw_pos, self.prev_realtime_pos, current_time, self.prev_realtime_time
                    )
                )
                
                # UI 文本刷新控制：每 3 帧更新一次数字，减少 PyQt 布局开销
                if self.frame_count % 3 == 0:
                    shot_count = self.trajectory_recorder.get_shot_count()
                    self.update_speed_display(speed, shot_count)

            # 3. 核心更新逻辑：直接提交 raw_pos
            self._process_ball_position_update(
                raw_pos, current_time, getattr(self, "_realtime_trajectory_index", 0), is_realtime=True
            )

            # 4. 渲染频率平衡：防止 OpenGL 刷新过快导致的主线程阻塞
            if hasattr(self, "plt") and self.plt:
                # addNewBall 仅添加数据点，更新非常快
                self.plt.addNewBall(raw_pos)
                # 渲染绘制：控制在约 60FPS 左右（假设数据源为 100Hz+，则隔帧绘制）
                if self.frame_count % 2 == 0:
                    self.plt.updatePlot()

            # 更新历史状态
            self.prev_realtime_pos = raw_pos.copy()
            self.prev_realtime_time = current_time
            self._realtime_trajectory_index = getattr(self, "_realtime_trajectory_index", 0) + 1

        except Exception as e:
            print(f"❌ 实时位置处理失败: {e}")

        def _analyze_realtime_landing(self, pos, current_time):
            """分析实时数据的落点

            Args:
                pos: 当前位置坐标
                current_time: 当前时间戳
            """
            try:
                # 使用落点分析模块进行实时落点分析
                landing_detected = self.landing_analyzer.analyze_realtime_landing(pos, current_time)
                
                # 如果检测到落点，更新图表显示（复用轨迹渲染模式的逻辑）
                if landing_detected:
                    print("🎯 实时落点检测完成，更新热力图和散点图")
                    
                    # 检查是否需要增加拍数（基于Y轴趋势变化）
                    if hasattr(self, "prev_realtime_pos") and hasattr(self, "prev_realtime_time") and \
                    self.prev_realtime_pos is not None and self.prev_realtime_time is not None:
                        
                        # 计算当前Y轴趋势
                        current_y = pos[1]
                        prev_y = self.prev_realtime_pos[1]
                        
                        if current_y is not None and prev_y is not None:
                            # 确定Y轴趋势
                            if current_y > prev_y:
                                current_y_trend = "上升"
                            elif current_y < prev_y:
                                current_y_trend = "下降"
                            else:
                                current_y_trend = "水平"
                            
                            # 检查趋势是否发生变化
                            if hasattr(self, "prev_realtime_y_trend") and \
                            self.prev_realtime_y_trend is not None and \
                            self.prev_realtime_y_trend != current_y_trend:
                                
                                # 趋势发生变化，增加拍数
                                shot_count = self.trajectory_recorder.get_shot_count()
                                print(f"🎯 实时模式检测到Y轴趋势变化: {self.prev_realtime_y_trend} -> {current_y_trend}")
                                print(f"📊 当前拍数: {shot_count}")
                            
                            # 更新前一个Y轴趋势
                            self.prev_realtime_y_trend = current_y_trend
                    
                    self.update_heatmap_display()
                    self.update_scatter_display()

            except Exception as e:
                print(f"❌ 实时落点分析失败: {e}")
                logger.error(f"Failed to analyze realtime landing: {str(e)}")



    def _analyze_realtime_landing(self, pos, current_time):
        """分析实时数据的落点

        Args:
            pos: 当前位置坐标
            current_time: 当前时间戳
        """
        try:
            # 使用落点分析模块进行实时落点分析
            landing_detected = self.landing_analyzer.analyze_realtime_landing(pos, current_time)
            
            # 如果检测到落点，更新图表显示（复用轨迹渲染模式的逻辑）
            if landing_detected:
                print("🎯 实时落点检测完成，更新热力图和散点图")
                
                # 检查是否需要增加拍数（基于Y轴趋势变化）
                if hasattr(self, "prev_realtime_pos") and hasattr(self, "prev_realtime_time") and \
                   self.prev_realtime_pos is not None and self.prev_realtime_time is not None:
                    
                    # 计算当前Y轴趋势
                    current_y = pos[1]
                    prev_y = self.prev_realtime_pos[1]
                    
                    if current_y is not None and prev_y is not None:
                        # 确定Y轴趋势
                        if current_y > prev_y:
                            current_y_trend = "上升"
                        elif current_y < prev_y:
                            current_y_trend = "下降"
                        else:
                            current_y_trend = "水平"
                        
                        # 检查趋势是否发生变化
                        if hasattr(self, "prev_realtime_y_trend") and \
                           self.prev_realtime_y_trend is not None and \
                           self.prev_realtime_y_trend != current_y_trend:
                            
                            # 趋势发生变化，增加拍数
                            shot_count = self.trajectory_recorder.get_shot_count()
                            print(f"🎯 实时模式检测到Y轴趋势变化: {self.prev_realtime_y_trend} -> {current_y_trend}")
                            print(f"📊 当前拍数: {shot_count}")
                        
                        # 更新前一个Y轴趋势
                        self.prev_realtime_y_trend = current_y_trend
                
                self.update_heatmap_display()
                self.update_scatter_display()

        except Exception as e:
            print(f"❌ 实时落点分析失败: {e}")
            logger.error(f"Failed to analyze realtime landing: {str(e)}")

    def _record_landing_point(self, timestamp, position):
        """记录落点坐标到统计中"""
        # 使用落点分析模块记录落点
        self.landing_analyzer.record_landing_point(timestamp, position)

        # 自动刷新热力图和散点图
        self.update_heatmap_display()
        self.update_scatter_display()

    def toggle_recording(self):
        """切换录制状态"""
        if not self.is_recording:
            # 开始录制
            self.start_recording()
        else:
            # 停止录制
            self.stop_recording()

    def start_recording(self):
        """开始录制视频"""
        try:
            # 获取窗口位置和大小
            window_geometry = self.main_widget.geometry()
            x, y = window_geometry.x(), window_geometry.y()
            width, height = window_geometry.width(), window_geometry.height()

            # 生成默认文件名和路径
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"pingpong_rec_{timestamp}.mp4"
            
            # 选择一个用户通常有权限的默认目录
            import os
            possible_dirs = [
                os.path.expanduser("~/Desktop"),      # 桌面
                os.path.expanduser("~/Documents"),    # 文档
                os.path.expanduser("~/Videos"),       # 视频文件夹
                os.path.expanduser("~/Downloads"),    # 下载文件夹
                os.path.expanduser("~"),              # 用户主目录
                os.getcwd(),                          # 当前工作目录
            ]
            
            # 找到第一个存在且可写的目录
            default_dir = os.getcwd()  # 备用默认值
            for dir_path in possible_dirs:
                if os.path.exists(dir_path) and os.access(dir_path, os.W_OK):
                    default_dir = dir_path
                    break
            
            default_full_path = os.path.join(default_dir, default_filename)

            # 打开文件保存对话框
            file_path, _ = QFileDialog.getSaveFileName(
                self.main_widget,
                "保存录制视频",
                default_full_path,
                "MP4 Files (*.mp4);;All Files (*)",
            )

            if file_path:
                try:
                    # 检查文件权限和目录访问
                    import os
                    import stat
                    
                    try:
                        # 检查目录是否存在和可写
                        dir_path = os.path.dirname(file_path)
                        if not os.path.exists(dir_path):
                            try:
                                os.makedirs(dir_path, exist_ok=True)
                                logger.info(f"创建目录: {dir_path}")
                            except Exception as mkdir_error:
                                logger.error(f"无法创建目录: {mkdir_error}")
                                QMessageBox.critical(
                                    self.main_widget, 
                                    "目录创建失败", 
                                    f"无法创建目录:\n{dir_path}\n\n错误: {mkdir_error}\n\n"
                                    "请选择其他位置或检查权限。"
                                )
                                return
                        
                        # 检查目录权限
                        if not os.access(dir_path, os.W_OK):
                            logger.error(f"目录无写入权限: {dir_path}")
                            QMessageBox.critical(
                                self.main_widget, 
                                "权限错误", 
                                f"目录没有写入权限:\n{dir_path}\n\n"
                                "解决方案:\n"
                                "1. 选择其他保存位置\n"
                                "2. 检查目录权限\n"
                                "3. 使用管理员权限运行程序"
                            )
                            return
                        
                        # 检查磁盘空间（估算需要至少100MB）
                        try:
                            statvfs = os.statvfs(dir_path)
                            free_space = statvfs.f_frsize * statvfs.f_bavail
                            required_space = 100 * 1024 * 1024  # 100MB
                            
                            if free_space < required_space:
                                logger.warning(f"磁盘空间不足: 可用 {free_space//1024//1024}MB")
                                reply = QMessageBox.question(
                                    self.main_widget,
                                    "磁盘空间警告",
                                    f"磁盘空间可能不足:\n"
                                    f"可用空间: {free_space//1024//1024}MB\n"
                                    f"建议空间: {required_space//1024//1024}MB\n\n"
                                    "是否继续录制？",
                                    QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No
                                )
                                if reply != QMessageBox.Yes:
                                    return
                        except Exception:
                            pass  # 忽略磁盘空间检查错误
                        
                        # 测试文件写入
                        with open(file_path, "wb") as f:
                            f.write(b"test")
                        os.remove(file_path)
                        
                    except PermissionError as e:
                        logger.error(f"权限错误: {e}")
                        QMessageBox.critical(
                            self.main_widget, 
                            "权限错误", 
                            f"没有文件写入权限:\n{file_path}\n\n"
                            "解决方案:\n"
                            "1. 选择其他保存位置（如桌面或文档文件夹）\n"
                            "2. 检查文件和目录权限\n"
                            "3. 关闭其他可能占用文件的程序"
                        )
                        return
                    except OSError as e:
                        logger.error(f"文件系统错误: {e}")
                        error_msg = str(e)
                        if "No space left" in error_msg:
                            error_detail = "磁盘空间不足"
                            solutions = "请清理磁盘空间或选择其他位置"
                        elif "Read-only" in error_msg:
                            error_detail = "文件系统为只读"
                            solutions = "请选择可写的位置"
                        else:
                            error_detail = f"文件系统错误: {error_msg}"
                            solutions = "请检查文件路径和权限"
                            
                        QMessageBox.critical(
                            self.main_widget, 
                            "文件系统错误", 
                            f"{error_detail}\n\n路径: {file_path}\n\n{solutions}"
                        )
                        return
                    except Exception as e:
                        logger.error(f"文件权限检查失败: {e}")
                        QMessageBox.critical(
                            self.main_widget, 
                            "文件访问错误", 
                            f"无法访问文件:\n{file_path}\n\n"
                            f"错误: {e}\n\n"
                            "请选择其他保存位置或检查权限。"
                        )
                        return

                    # 构建 FFmpeg 命令 - 根据操作系统选择合适的参数
                    import platform
                    import os
                    import re
                    system = platform.system()
                    
                    if system == "Linux":
                        # Linux 系统使用 x11grab 录制屏幕
                        # 自动检测正确的 DISPLAY
                        display = os.environ.get('DISPLAY', ':0')
                        if not display or display == '':
                            # 尝试检测系统中运行的 X 服务器
                            try:
                                # 查找 Xorg 进程和显示号
                                result = subprocess.run(['pgrep', '-a', 'Xorg'], capture_output=True, text=True)
                                if result.returncode == 0:
                                    # 从 Xorg 进程中提取显示号
                                    for line in result.stdout.split('\n'):
                                        if 'Xorg' in line and ':' in line:
                                            match = re.search(r':(\d+)', line)
                                            if match:
                                                display = f":{match.group(1)}"
                                                break
                                if display == ':0' or display == '':
                                    display = ':1'  # 常见的默认显示
                            except:
                                display = ':1'  # 回退到常见默认值
                        
                        screen_input = f"{display}+{window_geometry.x()},{window_geometry.y()}"
                        
                        ffmpeg_cmd = [
                            "ffmpeg",
                            "-f", "x11grab",  # X11 屏幕捕获
                            "-framerate", "30",  # 帧率
                            "-s", f"{window_geometry.width()}x{window_geometry.height()}",  # 窗口大小
                            "-i", screen_input,  # 屏幕位置（使用检测到的显示）
                            "-c:v", "libx264",  # 使用 libx264 编码器
                            "-preset", "fast",  # 编码预设
                            "-crf", "23",  # 质量设置 (18-28, 越小质量越好)
                            "-pix_fmt", "yuv420p",  # 像素格式
                            "-y",  # 覆盖已存在的文件
                            file_path,
                        ]
                        
                        logger.info(f"使用显示器: {display}")
                        logger.info(f"屏幕捕获区域: {screen_input}")
                    elif system == "Darwin":  # macOS
                        # macOS 系统使用 avfoundation
                        ffmpeg_cmd = [
                            "ffmpeg",
                            "-f", "avfoundation",  # 使用 macOS 的 avfoundation
                            "-framerate", "30",  # 帧率
                            "-i", "1:0",  # 输入设备（1 表示屏幕）
                            "-c:v", "h264_videotoolbox",  # 使用 VideoToolbox 硬件编码
                            "-b:v", "2000k",  # 视频比特率
                            "-pix_fmt", "yuv420p",  # 像素格式
                            "-y",  # 覆盖已存在的文件
                            file_path,
                        ]
                    else:
                        # 其他系统，默认使用 Linux 方式
                        logger.warning(f"未知操作系统: {system}，使用 Linux 录制方式")
                        ffmpeg_cmd = [
                            "ffmpeg",
                            "-f", "x11grab",  # 尝试 X11
                            "-framerate", "30",
                            "-s", f"{window_geometry.width()}x{window_geometry.height()}",
                            "-i", f":0.0+{window_geometry.x()},{window_geometry.y()}",
                            "-c:v", "libx264",
                            "-preset", "fast",
                            "-crf", "23",
                            "-pix_fmt", "yuv420p",
                            "-y",
                            file_path,
                        ]

                    # 环境检查和 FFmpeg 测试
                    if system == "Linux":
                        # 检查是否有图形环境运行
                        has_gui = False
                        display_to_test = os.environ.get('DISPLAY', '')
                        
                        # 如果没有 DISPLAY 环境变量，尝试检测系统中运行的 X 服务器
                        if not display_to_test:
                            try:
                                # 检查是否有 Xorg 进程运行
                                result = subprocess.run(['pgrep', 'Xorg'], capture_output=True)
                                if result.returncode == 0:
                                    # 有 X 服务器运行，尝试常见的显示号
                                    for test_display in [':1', ':0']:
                                        try:
                                            # 简单测试是否能连接到显示器
                                            test_result = subprocess.run(
                                                ['xdpyinfo', '-display', test_display], 
                                                capture_output=True, timeout=2
                                            )
                                            if test_result.returncode == 0:
                                                display_to_test = test_display
                                                has_gui = True
                                                logger.info(f"自动检测到显示器: {test_display}")
                                                break
                                        except:
                                            continue
                            except:
                                pass
                        else:
                            # 有 DISPLAY 环境变量，测试是否可用
                            try:
                                test_result = subprocess.run(['xdpyinfo'], capture_output=True, timeout=2)
                                has_gui = (test_result.returncode == 0)
                            except:
                                pass
                        
                        if not has_gui and not display_to_test:
                            logger.error("录制失败: 未检测到图形环境")
                            reply = QMessageBox.question(
                                self.main_widget, 
                                "图形环境检测失败", 
                                "无法检测到可用的图形界面环境\n\n"
                                "可能的原因：\n"
                                "• 系统没有启动图形界面\n"
                                "• X11 服务未运行\n"
                                "• 权限不足\n\n"
                                "解决方案：\n"
                                "1. 确保在图形桌面环境中运行\n"
                                "2. 检查显示管理器状态\n"
                                "3. 重启图形服务\n\n"
                                "是否导出数据文件代替录制？",
                                QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.Yes
                            )
                            if reply == QMessageBox.Yes:
                                self.export_current_data()
                            return
                        else:
                            logger.info(f"检测到图形环境，DISPLAY: {display_to_test or os.environ.get('DISPLAY', '未设置')}")
                    
                    logger.info(f"开始录制视频: {file_path}")
                    logger.info(f"操作系统: {system}")
                    logger.info(f"FFmpeg 命令: {' '.join(ffmpeg_cmd)}")

                    # 启动 FFmpeg 进程
                    self.ffmpeg_process = subprocess.Popen(
                        ffmpeg_cmd, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    
                    # 等待一小段时间检查进程是否成功启动
                    import time
                    time.sleep(0.5)
                    
                    if self.ffmpeg_process.poll() is not None:
                        # 进程已经退出，可能有错误
                        stdout, stderr = self.ffmpeg_process.communicate()
                        logger.error(f"FFmpeg 进程启动失败:")
                        logger.error(f"标准输出: {stdout}")
                        logger.error(f"错误输出: {stderr}")
                        
                        # 分析错误类型并提供针对性的解决方案
                        error_text = stderr.lower() if stderr else ""
                        
                        if "cannot open display" in error_text:
                            # X11 显示错误
                            reply = QMessageBox.question(
                                self.main_widget,
                                "显示器访问失败",
                                "屏幕录制失败：无法访问显示器\n\n"
                                "这通常发生在无图形界面的环境中：\n"
                                "• SSH 远程连接（未使用 -X）\n"
                                "• 服务器环境（无桌面）\n"
                                "• 容器环境（无图形支持）\n\n"
                                "解决方案：\n"
                                "1. 在本地桌面环境中运行\n"
                                "2. 使用 SSH -X 启用 X11 转发\n"
                                "3. 设置虚拟显示器 (Xvfb)\n\n"
                                "是否导出数据文件代替录制？",
                                QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.Yes
                            )
                            if reply == QMessageBox.Yes:
                                self.export_current_data()
                            return
                            
                        elif "permission denied" in error_text:
                            # 权限错误
                            QMessageBox.critical(
                                self.main_widget,
                                "权限错误",
                                "屏幕录制权限被拒绝\n\n"
                                "解决方案：\n"
                                "1. 检查应用程序录制权限\n"
                                "2. 使用管理员权限运行\n"
                                "3. 检查防火墙和安全软件设置"
                            )
                            return
                            
                        elif "no such file or directory" in error_text:
                            # 文件路径错误
                            QMessageBox.critical(
                                self.main_widget,
                                "路径错误",
                                "FFmpeg 找不到指定文件或设备\n\n"
                                "请检查：\n"
                                "1. FFmpeg 是否正确安装\n"
                                "2. 显示设备是否存在\n"
                                "3. 文件路径是否正确"
                            )
                            return
                            
                        elif "codec" in error_text or "encoder" in error_text:
                            # 编解码器错误
                            QMessageBox.critical(
                                self.main_widget,
                                "编码器错误",
                                f"FFmpeg 编码器问题\n\n"
                                f"可能原因：\n"
                                f"• 缺少必要的编解码器\n"
                                f"• 硬件编码不支持\n"
                                f"• FFmpeg 配置问题\n\n"
                                f"建议：\n"
                                f"1. 检查 FFmpeg 编译配置\n"
                                f"2. 尝试软件编码\n"
                                f"3. 更新 FFmpeg 版本"
                            )
                            return
                            
                        else:
                            # 通用错误处理
                            # 截取错误信息的关键部分
                            error_lines = stderr.split('\n') if stderr else []
                            key_errors = []
                            for line in error_lines:
                                if any(keyword in line.lower() for keyword in ['error', 'failed', 'cannot', 'unable']):
                                    key_errors.append(line.strip())
                            
                            error_summary = '\n'.join(key_errors[:3]) if key_errors else stderr[:300]
                            
                            reply = QMessageBox.question(
                                self.main_widget,
                                "录制启动失败",
                                f"FFmpeg 录制启动失败\n\n"
                                f"关键错误：\n{error_summary}\n\n"
                                f"常见解决方案：\n"
                                f"1. 检查系统图形环境\n"
                                f"2. 更新 FFmpeg 版本\n"
                                f"3. 检查权限设置\n\n"
                                f"是否导出数据文件代替录制？",
                                QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.Yes
                            )
                            if reply == QMessageBox.Yes:
                                self.export_current_data()
                            return

                    # 注册清理函数
                    atexit.register(self.cleanup_recording)

                    self.is_recording = True
                    self.record_btn.setText("Stop")
                    self.record_btn.setChecked(True)

                except Exception as e:
                    logger.error(f"启动录制时出错: {str(e)}")
                    QMessageBox.critical(
                        self.main_widget, "错误", f"启动录制时出错: {str(e)}"
                    )
                    self.cleanup_recording()
                    return

        except Exception as e:
            QMessageBox.critical(
                self.main_widget, "错误", f"创建视频文件时出错: {str(e)}"
            )
            self.cleanup_recording()
            return

    def stop_recording(self):
        """停止录制视频"""
        if self.is_recording and hasattr(self, "ffmpeg_process"):
            try:
                # 发送 SIGTERM 信号给 FFmpeg 进程
                self.ffmpeg_process.terminate()

                # 等待进程结束
                try:
                    self.ffmpeg_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # 如果进程没有及时结束，强制结束
                    self.ffmpeg_process.kill()

                self.is_recording = False
                self.record_btn.setText("Record")
                self.record_btn.setChecked(False)
                logger.info("停止录制视频")

            except Exception as e:
                logger.error(f"停止录制时出错: {str(e)}")
                QMessageBox.critical(
                    self.main_widget, "错误", f"停止录制时出错: {str(e)}"
                )

    def cleanup_recording(self):
        """清理录制相关的资源"""
        if hasattr(self, "ffmpeg_process"):
            try:
                if self.ffmpeg_process.poll() is None:  # 如果进程还在运行
                    self.ffmpeg_process.terminate()
                    try:
                        self.ffmpeg_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.ffmpeg_process.kill()
            except:
                pass
            self.ffmpeg_process = None

        self.is_recording = False
        if hasattr(self, "record_btn"):
            self.record_btn.setText("Record")
            self.record_btn.setChecked(False)

    def toggle_realtime_render(self):
        """切换实时渲染模式"""
        if not hasattr(self, 'realtime_render_btn'):
            return
            
        if self.realtime_render_btn.isChecked():
            self.start_realtime_render()
        else:
            self.stop_realtime_render()

    def start_realtime_render(self):
        """启动实时渲染模式"""
        try:
            # 检查LCM可用性
            if not LCM_AVAILABLE:
                print("❌ LCM库不可用，无法启动实时渲染")
                self.realtime_render_btn.setChecked(False)
                QMessageBox.warning(
                    self.main_widget, 
                    "警告", 
                    "LCM库不可用，无法启动实时渲染模式"
                )
                return

            # 检查LCM数据可用性
            if not self._check_lcm_data_availability():
                print("⚠️ 未检测到LCM数据，无法启动实时渲染")
                self.realtime_render_btn.setChecked(False)
                QMessageBox.warning(
                    self.main_widget, 
                    "警告", 
                    "未检测到LCM数据，请确保数据发送方已启动"
                )
                return

            # 启动LCM订阅
            self.start_lcm_subscription()
            
            # 切换到实时模式
            self.switch_to_real_time_mode()
            
            # 更新按钮状态
            self.realtime_render_btn.setText("Stop Real-time")
            self.realtime_render_btn.setChecked(True)
            
            print("✅ 实时渲染模式已启动")
            
        except Exception as e:
            print(f"❌ 启动实时渲染模式失败: {e}")
            self.realtime_render_btn.setChecked(False)
            QMessageBox.critical(
                self.main_widget, 
                "错误", 
                f"启动实时渲染模式失败: {str(e)}"
            )

    def stop_realtime_render(self):
        """停止实时渲染模式"""
        try:
            # 停止LCM订阅
            self.stop_lcm_subscription()
            
            # 更新按钮状态
            self.realtime_render_btn.setText("Real-time Render")
            self.realtime_render_btn.setChecked(False)
            
            print("✅ 实时渲染模式已停止")
            
        except Exception as e:
            print(f"❌ 停止实时渲染模式失败: {e}")
            QMessageBox.critical(
                self.main_widget, 
                "错误", 
                f"停止实时渲染模式失败: {str(e)}"
            )

    def _check_lcm_data_availability(self):
        """检查LCM数据可用性"""
        try:
            if not LCM_AVAILABLE:
                return False
                
            # 创建临时LCM实例进行检测
            test_lcm = lcm.LCM()
            
            # 设置短超时检测是否有数据
            message_count = test_lcm.handle_timeout(500)  # 500ms超时
            
            # 清理临时实例
            del test_lcm
            
            # 如果收到消息或者没有错误，认为LCM可用
            return message_count >= 0
            
        except Exception as e:
            print(f"⚠️ 检查LCM数据可用性时出错: {e}")
            return False

    def _init_realtime_render_button_state(self):
        """初始化实时渲染按钮状态"""
        try:
            if not hasattr(self, 'realtime_render_btn'):
                return
                
            # 检查LCM库可用性
            if not LCM_AVAILABLE:
                self.realtime_render_btn.setEnabled(False)
                self.realtime_render_btn.setText("Real-time Render (LCM Not Available)")
                print("⚠️ LCM库不可用，实时渲染按钮已禁用")
                return
                
            # 检查LCM数据可用性
            if self._check_lcm_data_availability():
                self.realtime_render_btn.setEnabled(True)
                print("✅ 检测到LCM数据，实时渲染功能可用")
                
                # 默认启动实时渲染模式
                try:
                    print("🚀 默认启动实时渲染模式...")
                    self.realtime_render_btn.setChecked(True)
                    self.start_realtime_render()
                except Exception as e:
                    print(f"⚠️ 默认启动实时渲染失败: {e}")
                    self.realtime_render_btn.setChecked(False)
                    self.realtime_render_btn.setText("Real-time Render")
            else:
                self.realtime_render_btn.setEnabled(False)
                self.realtime_render_btn.setText("Real-time Render (No Data)")
                print("⚠️ 未检测到LCM数据，实时渲染按钮已禁用")
                
        except Exception as e:
            print(f"❌ 初始化实时渲染按钮状态失败: {e}")
            if hasattr(self, 'realtime_render_btn'):
                self.realtime_render_btn.setEnabled(False)
                self.realtime_render_btn.setText("Real-time Render (Error)")

    def reset_chart_data(self):
        """重置图表数据，清理当前存档中的速度与落点数据"""
        try:
            # 显示确认对话框
            reply = QMessageBox.question(
                self.main_widget,
                "确认重置",
                "确定要清空当前存档中的所有速度数据和落点数据吗？\n\n文件将被保留，但内容会被清空并重置为初始状态！",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                print("🚫 用户取消了重置操作")
                return
                
            # 确定数据目录
            if self.save_folder_path:
                base_dir = self.save_folder_path
            else:
                base_dir = "."
                
            # 清理速度数据
            speed_data_dir = os.path.join(base_dir, "speed_data")
            if os.path.exists(speed_data_dir):
                speed_file = os.path.join(speed_data_dir, "speed_data.csv")
                if os.path.exists(speed_file):
                    try:
                        # 清空文件内容并写入表头
                        with open(speed_file, 'w', encoding='utf-8', newline='') as f:
                            import csv
                            writer = csv.writer(f)
                            # 写入CSV表头（与trajectory_recorder.py中的表头保持一致）
                            writer.writerow([
                                "timestamp",
                                "frame_count",
                                "speed_mps",
                                "y_trend",
                                "y_trend_changed",
                                "player_side",
                                "x_mm",
                                "y_mm",
                                "z_mm",
                                "prev_x_mm",
                                "prev_y_mm",
                                "prev_z_mm",
                                "shot_count",
                            ])
                        print(f"✅ 已清空速度数据文件: {speed_file}")
                    except Exception as e:
                        print(f"⚠️ 清空速度数据文件失败: {e}")
                        
            # 清理落点数据
            landing_data_dir = os.path.join(base_dir, "landing_data")
            if os.path.exists(landing_data_dir):
                landing_file = os.path.join(landing_data_dir, "landing_data.csv")
                if os.path.exists(landing_file):
                    try:
                        # 清空文件内容并写入表头
                        with open(landing_file, 'w', encoding='utf-8', newline='') as f:
                            import csv
                            writer = csv.writer(f)
                            # 写入CSV表头（与landing_analyzer.py中的表头保持一致）
                            writer.writerow([
                                "timestamp",
                                "frame_count",
                                "x_mm",
                                "y_mm",
                                "z_mm",
                                "intensity",
                                "bin_x",
                                "bin_y",
                                "distance_from_last",
                            ])
                        print(f"✅ 已清空落点数据文件: {landing_file}")
                    except Exception as e:
                        print(f"⚠️ 清空落点数据文件失败: {e}")
                        
            # 清理轨迹数据
            trajectory_data_dir = os.path.join(base_dir, "trajectory_data")
            if os.path.exists(trajectory_data_dir):
                trajectory_file = os.path.join(trajectory_data_dir, "trajectory_data.csv")
                if os.path.exists(trajectory_file):
                    try:
                        # 清空文件内容并写入表头
                        with open(trajectory_file, 'w', encoding='utf-8', newline='') as f:
                            import csv
                            writer = csv.writer(f)
                            # 写入CSV表头（与trajectory_recorder.py中的表头保持一致）
                            writer.writerow([
                                "timestamp",
                                "frame_count",
                                "x_mm",
                                "y_mm",
                                "z_mm",
                                "is_original_point",
                                "is_interpolated_point",
                                "is_landing_point",
                            ])
                        print(f"✅ 已清空轨迹数据文件: {trajectory_file}")
                    except Exception as e:
                        print(f"⚠️ 清空轨迹数据文件失败: {e}")
                        
            # 刷新图表显示
            self._refresh_charts()
            
            # 显示成功消息
            QMessageBox.information(
                self.main_widget,
                "重置完成",
                "图表数据已成功清空！\n\n所有速度数据、落点数据和轨迹数据已重置为初始状态。"
            )
            
            print("🎯 图表数据重置完成")
            
        except Exception as e:
            error_msg = f"重置图表数据失败: {str(e)}"
            print(f"❌ {error_msg}")
            QMessageBox.critical(
                self.main_widget,
                "错误",
                error_msg
            )

    def _refresh_charts(self):
        """刷新所有图表显示"""
        try:
            # 刷新热力图
            if hasattr(self, 'heatmap_canvas'):
                self.heatmap_canvas.setText("No data available")
                self.heatmap_canvas.setAlignment(Qt.AlignCenter)
                
            # 刷新散点图
            if hasattr(self, 'scatter_canvas'):
                self.scatter_canvas.setText("No data available")
                self.scatter_canvas.setAlignment(Qt.AlignCenter)
                
            # 刷新速度图表 - 调用update_speed_chart方法而不是直接设置文本
            if hasattr(self, 'speed_chart_label'):
                try:
                    self.update_speed_chart()
                    print("✅ 速度图表已刷新")
                except Exception as e:
                    print(f"⚠️ 刷新速度图表失败: {e}")
                    # 作为后备方案，直接设置文本
                    self.speed_chart_label.setText("No data available")
                    self.speed_chart_label.setAlignment(Qt.AlignCenter)
                
            # 重置速度显示标签
            if hasattr(self, 'speed_label'):
                self.speed_label.setText("time: 00:00:00\nSpeed: 0.0 m/s\nShots: 0")
                
            print("🔄 图表显示已刷新")
            
        except Exception as e:
            print(f"⚠️ 刷新图表显示失败: {e}")

    # 数据记录初始化方法已移至相应的模块中

    def record_trajectory_data_point(self, pos):
        """记录轨迹数据点到trajectory_data文件"""
        try:
            # 使用轨迹记录模块记录轨迹数据
            # 检查是否有有效的轨迹数据
            if (hasattr(self, 'complete_trajectory') and 
                len(self.complete_trajectory) > 0 and 
                hasattr(self, 'trajectory_index') and 
                0 <= self.trajectory_index < len(self.complete_trajectory)):
                current_data = self.complete_trajectory[self.trajectory_index]
                self.trajectory_recorder.record_trajectory_data_point(pos, current_data)
            else:
                # 实时模式下没有轨迹数据，创建一个简单的数据结构
                current_time = getattr(self, 'current_time', time.time())
                frame_count = getattr(self, 'frame_count', 0)
                current_data = {
                    "position": pos,
                    "time": current_time,
                    "frame": frame_count
                }
                self.trajectory_recorder.record_trajectory_data_point(pos, current_data)
                
        except Exception as e:
            print(f"❌ 记录轨迹数据点失败: {e}")
            logger.error(f"Failed to record trajectory data point: {str(e)}")

    def get_heatmap_data(self):
        """获取热力图数据供界面显示 - 从landing_data文件加载累积落点"""
        return self.chart_renderer.get_heatmap_data()

    def get_scatter_data(self):
        """获取散点图数据 - 从文件加载累积数据"""
        return self.chart_renderer.get_scatter_data()

    # 数据记录相关方法已移至相应的模块中

    def update_heatmap_display(self):
        """更新热力图显示 - 从文件加载累积数据"""
        try:
            heatmap_data = self.get_heatmap_data()

            if heatmap_data[0] is not None and np.max(heatmap_data[0]) > 0:
                print(f"✅ 加载热力图数据，最大落点数: {np.max(heatmap_data[0])}")
                self.draw_heatmap_plot(heatmap_data)
            else:
                print("⚠️ 热力图数据为空")
                self.heatmap_canvas.setText(
                    "No landing data\nPlease run simulator to record landing points"
                )
        except Exception as e:
            print(f"❌ 更新热力图显示时出错: {str(e)}")
            self.heatmap_canvas.setText(f"Heatmap display error: {str(e)}")

    def update_scatter_display(self):
        """更新散点图显示 - 从文件加载累积数据"""
        try:
            scatter_data = self.get_scatter_data()

            if scatter_data and len(scatter_data) > 0:
                print(f"✅ 加载散点图数据，落点数: {len(scatter_data)}")
                self.draw_scatter_plot(scatter_data)
            else:
                print("⚠️ 散点图数据为空")
                self.scatter_canvas.setText(
                    "No landing data\nPlease run simulator to record landing points"
                )
        except Exception as e:
            print(f"❌ 更新散点图显示时出错: {str(e)}")
            self.scatter_canvas.setText(f"Scatter display error: {str(e)}")

    def draw_heatmap_plot(self, heatmap_data):
        """绘制热力图"""
        self.chart_renderer.draw_heatmap_plot(heatmap_data, self.heatmap_canvas)

    def draw_scatter_plot(self, scatter_data):
        """绘制散点图"""
        self.chart_renderer.draw_scatter_plot(scatter_data, self.scatter_canvas)

    def update_speed_chart(self):
        """更新速度折线图显示"""
        try:
            print("🔄 开始更新速度图表...")
            
            # 使用图表渲染模块获取速度数据并绘制
            speed_data = self.chart_renderer.get_speed_chart_data()
            print(f"📊 获取到速度数据: {speed_data}")
            
            blue_speeds, blue_shot_numbers, green_speeds, green_shot_numbers = speed_data
            
            # 检查是否有有效数据（不是None且不是空列表）
            has_blue_data = blue_speeds is not None and len(blue_speeds) > 0
            has_green_data = green_speeds is not None and len(green_speeds) > 0
            
            print(f"📊 蓝方数据: {has_blue_data}, 绿方数据: {has_green_data}")
            
            if has_blue_data or has_green_data:
                print("✅ 有有效数据，开始绘制速度图表...")
                self.chart_renderer.draw_speed_chart(speed_data, self.speed_chart_label)
                print("✅ 速度图表绘制完成")
            else:
                print("⚠️ 没有有效数据，显示无数据提示")
                self.speed_chart_label.setText("No valid speed data")
                self.speed_chart_label.setAlignment(Qt.AlignCenter)
                
        except Exception as e:
            print(f"❌ 更新速度图表失败: {e}")
            import traceback
            traceback.print_exc()
            # 作为后备方案，显示错误信息
            self.speed_chart_label.setText("Error loading speed data")
            self.speed_chart_label.setAlignment(Qt.AlignCenter)

    # 插值相关方法已移至 interpolation.py 模块

    def _analyze_landing_from_trajectory(self, trajectory_index):
        """从完整轨迹队列中分析落点：第一次检测z<60时标记疑似落点，第二次检测z上升时确认落点"""
        # 使用落点分析模块进行分析
        landing_detected = self.landing_analyzer.analyze_landing_from_trajectory(
            trajectory_index, self.complete_trajectory
        )

        # 如果检测到落点，更新图表显示
        if landing_detected:
            print("🎯 落点检测完成，更新热力图和散点图")
            self.update_heatmap_display()
            self.update_scatter_display()

    def cleanup(self):
        """清理资源."""
        if self.is_recording:
            self.stop_recording()

        # 关闭采集程序进程 - 使用简洁的endprocess方式
        self._force_kill_collection_process()

    def _force_kill_collection_process(self):
        """强制终止采集程序进程 - 类似系统监视器的endprocess"""
        if not hasattr(self, "collection_process") or not self.collection_process:
            return
            
        try:
            pid = self.collection_process.pid
            print(f"🔄 强制终止采集程序进程 (PID: {pid})")
            
            # 直接使用kill()强制终止，类似系统监视器的endprocess
            self.collection_process.kill()
            
            # 等待进程结束，但不等太久
            try:
                self.collection_process.wait(timeout=1)
                print("✅ 采集程序进程已强制终止")
            except subprocess.TimeoutExpired:
                # 如果1秒内没有结束，使用系统级强制终止
                import os, signal
                try:
                    os.kill(pid, signal.SIGKILL)
                    print("✅ 使用系统级SIGKILL强制终止")
                except (ProcessLookupError, OSError):
                    print("✅ 进程已不存在")
            
            # 清空进程引用
            self.collection_process = None
            
        except Exception as e:
            print(f"⚠️ 强制终止采集程序失败: {e}")
            # 最后尝试系统级清理
            self._cleanup_all_trajectory_simulators()

    def _cleanup_all_trajectory_simulators(self):
        """系统级清理所有轨迹模拟器进程"""
        try:
            import psutil
            
            # 查找所有trajectory_sender进程
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] and 'trajectory_sender' in proc.info['name']:
                        print(f"🔍 发现轨迹模拟器进程: PID={proc.info['pid']}, 名称={proc.info['name']}")
                        proc.kill()  # 直接使用kill()强制终止
                        print(f"✅ 已强制终止进程 PID={proc.info['pid']}")
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                    pass
                    
        except ImportError:
            print("⚠️ psutil库不可用，跳过系统级进程清理")
        except Exception as e:
            print(f"⚠️ 系统级进程清理失败: {e}")
        
        # 清理可能存在的终端窗口
        self._cleanup_terminal_windows()

        # 关闭落点数据记录
        self.landing_analyzer.close_landing_data_recording()

        # 关闭球速数据记录
        self.trajectory_recorder.close_speed_data_recording()

        # 关闭轨迹数据记录
        self.trajectory_recorder.close_trajectory_data_recording()

        # 停止训练定时器
        if hasattr(self, "training_timer") and self.training_timer.isActive():
            self.training_timer.stop()
            print("⏹️ 训练定时器已停止")

        # 清理3D可视化资源
        if hasattr(self, "plt") and hasattr(self.plt, "cleanup"):
            self.plt.cleanup()

        try:
            logger.info("Simulation completed and resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise

    def _cleanup_all_trajectory_simulators(self):
        """系统级清理所有轨迹模拟器进程"""
        try:
            print("🔄 执行系统级轨迹模拟器进程清理...")
            
            # 1. 清理轨迹模拟器主进程
            result = subprocess.run(['pgrep', '-f', 'trajectory_simulator'], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0 and result.stdout.strip():
                pids = [pid.strip() for pid in result.stdout.strip().split('\n') if pid.strip()]
                print(f"🔍 找到 {len(pids)} 个轨迹模拟器进程: {pids}")
                
                # 逐个清理进程
                for pid_str in pids:
                    try:
                        pid = int(pid_str)
                        print(f"🔄 清理进程 PID: {pid}")
                        
                        # 发送 SIGTERM
                        os.kill(pid, signal.SIGTERM)
                        
                        # 等待进程结束
                        for i in range(5):  # 等待最多5秒
                            try:
                                os.kill(pid, 0)  # 检查进程是否还存在
                                time.sleep(1)
                            except ProcessLookupError:
                                print(f"✅ 进程 {pid} 已正常关闭")
                                break
                        else:
                            # 强制终止
                            try:
                                os.kill(pid, signal.SIGKILL)
                                print(f"✅ 进程 {pid} 已强制关闭")
                            except ProcessLookupError:
                                print(f"🔍 进程 {pid} 已不存在")
                                
                    except (ValueError, ProcessLookupError):
                        print(f"🔍 进程 {pid_str} 已不存在或无效")
            
            # 2. 清理可能遗留的 shell 进程（由 shell=True 创建）
            print("🔄 检查并清理相关的 shell 进程...")
            try:
                shell_result = subprocess.run(['pgrep', '-f', 'sh.*trajectory_simulator'], 
                                            capture_output=True, text=True, timeout=3)
                if shell_result.returncode == 0 and shell_result.stdout.strip():
                    shell_pids = [pid.strip() for pid in shell_result.stdout.strip().split('\n') if pid.strip()]
                    print(f"🔍 找到 {len(shell_pids)} 个相关 shell 进程: {shell_pids}")
                    
                    for pid_str in shell_pids:
                        try:
                            pid = int(pid_str)
                            os.kill(pid, signal.SIGTERM)
                            print(f"🔄 清理 shell 进程 PID: {pid}")
                        except (ValueError, ProcessLookupError):
                            pass
            except Exception as e:
                print(f"🔍 shell 进程检查: {e}")
            
            # 3. 清理可能的终端进程
            print("🔄 检查并清理可能的终端进程...")
            try:
                # 方法1: 查找包含 trajectory_simulator 相关的终端进程
                import platform
                system = platform.system()
                
                if system == "Darwin":  # macOS
                    terminal_patterns = [
                        'Terminal.*trajectory_simulator',
                        'iTerm.*trajectory_simulator',
                        'iTerm2.*trajectory_simulator',
                        'Hyper.*trajectory_simulator',
                        'Alacritty.*trajectory_simulator'
                    ]
                else:  # Linux 和其他系统
                    terminal_patterns = [
                        'gnome-terminal.*trajectory_simulator',
                        'xterm.*trajectory_simulator', 
                        'konsole.*trajectory_simulator',
                        'terminal.*trajectory_simulator',
                        'terminator.*trajectory_simulator',
                        'tilix.*trajectory_simulator'
                    ]
                
                for pattern in terminal_patterns:
                    try:
                        term_result = subprocess.run(['pgrep', '-f', pattern], 
                                                   capture_output=True, text=True, timeout=2)
                        if term_result.returncode == 0 and term_result.stdout.strip():
                            term_pids = [pid.strip() for pid in term_result.stdout.strip().split('\n') if pid.strip()]
                            print(f"🔍 找到终端进程 ({pattern}): {term_pids}")
                            
                            for pid_str in term_pids:
                                try:
                                    pid = int(pid_str)
                                    os.kill(pid, signal.SIGTERM)
                                    print(f"🔄 清理终端进程 PID: {pid}")
                                except (ValueError, ProcessLookupError):
                                    pass
                    except Exception:
                        continue
                
                # 方法2: 查找可能通过终端启动但现在已孤立的终端窗口
                print("🔄 检查孤立的终端窗口...")
                try:
                    # 获取当前用户的所有终端进程
                    # 根据操作系统使用不同的终端应用名称
                    import platform
                    system = platform.system()
                    
                    if system == "Darwin":  # macOS
                        # macOS 常见终端应用
                        terminal_pattern = '(Terminal|iTerm|iTerm2|Hyper|Alacritty)'
                    else:  # Linux 和其他系统
                        terminal_pattern = '(gnome-terminal|xterm|konsole|terminator|tilix)'
                    
                    user_terminals = subprocess.run(
                        ['pgrep', '-u', str(os.getuid()), '-f', terminal_pattern], 
                        capture_output=True, text=True, timeout=3
                    )
                    
                    if user_terminals.returncode == 0 and user_terminals.stdout.strip():
                        terminal_pids = [pid.strip() for pid in user_terminals.stdout.strip().split('\n') if pid.strip()]
                        print(f"🔍 找到用户终端进程: {terminal_pids}")
                        
                        # 检查这些终端是否可能与 trajectory_simulator 相关
                        for pid_str in terminal_pids:
                            try:
                                pid = int(pid_str)
                                # 检查进程的命令行参数和子进程
                                # macOS 和 Linux 的 ps 命令参数略有不同
                                import platform
                                system = platform.system()
                                
                                if system == "Darwin":  # macOS
                                    ps_cmd = ['ps', '-p', str(pid), '-o', 'command']
                                else:  # Linux 和其他系统
                                    ps_cmd = ['ps', '-p', str(pid), '-o', 'cmd', '--no-headers']
                                
                                cmdline_result = subprocess.run(
                                    ps_cmd, 
                                    capture_output=True, text=True, timeout=2
                                )
                                
                                if cmdline_result.returncode == 0:
                                    cmdline = cmdline_result.stdout.strip()
                                    # 如果终端标题或命令行包含 trajectory_simulator 相关信息
                                    if any(keyword in cmdline.lower() for keyword in ['trajectory', 'simulator', 'pingpong']):
                                        print(f"🔍 发现可能相关的终端: PID {pid} - {cmdline}")
                                        
                                        # 友好提示而不是直接关闭，因为可能是用户手动打开的
                                        print(f"ℹ️ 发现可能相关的终端窗口 (PID: {pid})")
                                        print(f"   如果这是手动打开的终端，请手动关闭")
                                        
                            except (ValueError, ProcessLookupError, subprocess.TimeoutExpired):
                                continue
                                
                except Exception as e:
                    print(f"🔍 孤立终端检查失败: {e}")
                    
            except Exception as e:
                print(f"🔍 终端进程检查: {e}")
                
                # 最终验证
                time.sleep(1)
                verify_result = subprocess.run(['pgrep', '-f', 'trajectory_simulator'], 
                                             capture_output=True, text=True, timeout=3)
                if verify_result.returncode == 0:
                    remaining = verify_result.stdout.strip().split('\n')
                    print(f"⚠️ 还有 {len(remaining)} 个进程未清理: {remaining}")
                    # 使用 pkill 作为最后手段
                    subprocess.run(['pkill', '-9', '-f', 'trajectory_simulator'], 
                                 capture_output=True, timeout=3)
                    print("✅ 已使用 pkill -9 强制清理")
                else:
                    print("✅ 所有轨迹模拟器进程已清理完毕")
                    
            else:
                print("🔍 没有找到需要清理的轨迹模拟器进程")
                
        except Exception as e:
            print(f"❌ 系统级进程清理失败: {e}")
            # 最后的最后：直接使用 pkill
            try:
                subprocess.run(['pkill', '-9', '-f', 'trajectory_simulator'], 
                             capture_output=True, timeout=3)
                print("✅ 已使用 pkill -9 作为最后手段清理")
            except Exception as final_error:
                print(f"❌ 最终清理也失败: {final_error}")
    
    def _cleanup_terminal_windows(self):
        """清理可能存在的终端窗口"""
        try:
            print("🔄 检查并清理终端窗口...")
            
            # 方法1: 使用平台特定的窗口管理工具
            import platform
            system = platform.system()
            
            try:
                if system == "Darwin":  # macOS
                    # macOS 使用 AppleScript 查找和关闭窗口
                    print("🔍 使用 AppleScript 查找相关终端窗口...")
                    try:
                        # 查找 Terminal 应用中包含相关内容的窗口
                        applescript_cmd = '''
                        tell application "Terminal"
                            repeat with w in windows
                                try
                                    set windowName to name of w
                                    if windowName contains "trajectory" or windowName contains "simulator" or windowName contains "pingpong" then
                                        close w
                                        return "Closed window: " & windowName
                                    end if
                                end try
                            end repeat
                        end tell
                        '''
                        
                        result = subprocess.run(['osascript', '-e', applescript_cmd], 
                                              capture_output=True, text=True, timeout=5)
                        
                        if result.returncode == 0 and result.stdout.strip():
                            print(f"✅ {result.stdout.strip()}")
                        
                    except Exception as e:
                        print(f"🔍 AppleScript 终端窗口管理失败: {e}")
                        
                else:  # Linux 和其他系统
                    # 检查是否有 wmctrl 工具
                    wmctrl_check = subprocess.run(['which', 'wmctrl'], 
                                                capture_output=True, text=True, timeout=2)
                    
                    if wmctrl_check.returncode == 0:
                        print("🔍 使用 wmctrl 查找相关终端窗口...")
                        
                        # 查找包含 trajectory_simulator 的窗口
                        window_list = subprocess.run(['wmctrl', '-l'], 
                                                   capture_output=True, text=True, timeout=3)
                        
                        if window_list.returncode == 0:
                            for line in window_list.stdout.split('\n'):
                                if line.strip() and any(keyword in line.lower() for keyword in 
                                                      ['trajectory', 'simulator', 'pingpong']):
                                    print(f"🔍 发现相关窗口: {line.strip()}")
                                    # 提取窗口ID
                                    window_id = line.split()[0]
                                    try:
                                        subprocess.run(['wmctrl', '-ic', window_id], 
                                                     capture_output=True, timeout=2)
                                        print(f"✅ 已关闭窗口: {window_id}")
                                    except Exception as e:
                                        print(f"⚠️ 关闭窗口失败: {e}")
                    else:
                        print("🔍 wmctrl 不可用，跳过窗口管理")
                    
            except Exception as e:
                print(f"🔍 窗口管理工具检查失败: {e}")
            
            # 方法2: 检查当前桌面环境的终端管理器
            try:
                desktop_env = os.environ.get('DESKTOP_SESSION', '').lower()
                xdg_current_desktop = os.environ.get('XDG_CURRENT_DESKTOP', '').lower()
                
                print(f"🔍 检测到桌面环境: {desktop_env}, {xdg_current_desktop}")
                
                # 对于 GNOME 环境
                if 'gnome' in desktop_env or 'gnome' in xdg_current_desktop:
                    try:
                        # 尝试通过 dbus 获取 gnome-terminal 信息
                        result = subprocess.run([
                            'gdbus', 'call', '--session', 
                            '--dest', 'org.gnome.Terminal',
                            '--object-path', '/org/gnome/Terminal/Factory0',
                            '--method', 'org.gtk.Application.ListActions'
                        ], capture_output=True, text=True, timeout=3)
                        
                        if result.returncode == 0:
                            print("🔍 检测到 GNOME Terminal 服务")
                            print("ℹ️ 请手动关闭包含 trajectory_simulator 的终端窗口")
                        
                    except Exception as e:
                        print(f"🔍 GNOME Terminal 检查: {e}")
                
                # 通用方法：提醒用户
                print("ℹ️ 提醒：如果仍有终端窗口显示 trajectory_simulator 相关内容，")
                print("   请手动关闭这些窗口以完全清理环境")
                print("   💡 小贴士：可以使用 Ctrl+C 或直接关闭终端窗口")
                
            except Exception as e:
                print(f"🔍 桌面环境检查失败: {e}")
                
        except Exception as e:
            print(f"❌ 终端窗口清理失败: {e}")

    def export_current_data(self):
        """导出当前数据到文件"""
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 打开文件保存对话框
            default_filename = f"pingpong_data_export_{timestamp}.json"
            file_path, _ = QFileDialog.getSaveFileName(
                self.main_widget,
                "导出数据",
                default_filename,
                "JSON Files (*.json);;CSV Files (*.csv);;All Files (*)",
            )
            
            if file_path:
                # 收集当前数据
                export_data = {
                    "timestamp": timestamp,
                    "trajectory_points": len(self.trajectory_points) if hasattr(self, 'trajectory_points') else 0,
                    "speed_data": [],
                    "landing_points": [],
                    "statistics": {}
                }
                
                # 导出轨迹点数据
                if hasattr(self, 'trajectory_points') and self.trajectory_points:
                    export_data["trajectory_data"] = [
                        {
                            "timestamp": point[0] if isinstance(point, (list, tuple)) else point.get('timestamp', 0),
                            "x": point[1] if isinstance(point, (list, tuple)) else point.get('x', 0),
                            "y": point[2] if isinstance(point, (list, tuple)) else point.get('y', 0),
                            "z": point[3] if isinstance(point, (list, tuple)) else point.get('z', 0)
                        }
                        for point in self.trajectory_points[:1000]  # 限制导出前1000个点
                    ]
                
                # 导出速度数据
                if hasattr(self, 'speed_data') and self.speed_data:
                    export_data["speed_data"] = self.speed_data[-100:]  # 最近100个速度数据
                
                # 导出落点数据
                if hasattr(self, 'landing_analyzer') and hasattr(self.landing_analyzer, 'landing_points'):
                    export_data["landing_points"] = [
                        {"x": point[0], "y": point[1]} 
                        for point in self.landing_analyzer.landing_points[-100:]  # 最近100个落点
                    ]
                
                # 导出统计数据
                if hasattr(self, 'training_start_time'):
                    export_data["statistics"] = {
                        "training_duration": self.calculate_training_time() if self.training_start_time else 0,
                        "total_shots": len(export_data.get("landing_points", [])),
                        "export_time": timestamp
                    }
                
                # 写入文件
                import json
                if file_path.endswith('.json'):
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(export_data, f, indent=2, ensure_ascii=False)
                elif file_path.endswith('.csv'):
                    # CSV 格式导出轨迹数据
                    import csv
                    with open(file_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(['timestamp', 'x', 'y', 'z'])
                        for point in export_data.get("trajectory_data", []):
                            writer.writerow([point['timestamp'], point['x'], point['y'], point['z']])
                
                logger.info(f"数据导出成功: {file_path}")
                QMessageBox.information(
                    self.main_widget,
                    "导出成功",
                    f"数据已成功导出到:\n{file_path}\n\n"
                    f"包含数据:\n"
                    f"• 轨迹点: {len(export_data.get('trajectory_data', []))}\n"
                    f"• 速度数据: {len(export_data.get('speed_data', []))}\n"
                    f"• 落点数据: {len(export_data.get('landing_points', []))}"
                )
                
        except Exception as e:
            logger.error(f"数据导出失败: {e}")
            QMessageBox.critical(
                self.main_widget,
                "导出错误",
                f"数据导出失败:\n{str(e)}"
            )

    def resizeEvent(self, event):
        """处理窗口大小改变事件"""
        super().resizeEvent(event)

        print(
            f"🔄 窗口大小改变事件触发: {event.size().width()}x{event.size().height()}"
        )

        # 等待窗口大小调整完成
        QTimer.singleShot(100, self._update_ui_positions)

        # 更新球台大小
        self.update_table_size()
        # 更新视图
        self.update()

    def _update_ui_positions(self):
        """更新UI元素位置"""
        try:
            # 获取当前窗口尺寸
            window_width = self.main_widget.width()
            window_height = self.main_widget.height()

            print(f"🔄 更新UI位置 - 窗口尺寸: {window_width}x{window_height}")

            # 定义边距
            margin = 30
            chart_height = 300

            # 1. 左侧按钮区域（左上角）
            self.local_monitor_btn.move(margin, margin)
            self.local_trajectory_btn.move(margin, margin + 50)
            self.record_btn.move(margin, margin + 100)
            self.realtime_render_btn.move(margin, margin + 150)
            self.reset_charts_btn.move(margin, margin + 200)

            # [新增] 发球评估按钮的位置 (在 reset_charts_btn 下方 50px)
            if hasattr(self, 'eval_serve_btn'):
                self.eval_serve_btn.move(margin, margin + 250)

            # 2. 右侧控制区域（右上角）
            # 按钮框架
            self.button_frame.move(
                window_width - self.button_frame.width() - margin,
                window_height - self.button_frame.height() - margin,
            )

            # 球速标签（屏幕中间，距离上边80px）
            speed_label_x = (window_width - self.speed_label.width()) // 2
            self.speed_label.move(speed_label_x, 80)
            print(f"📍 球速标签位置: ({speed_label_x}, 80), 宽度: {self.speed_label.width()}")

            # 速度折线图（屏幕右上方，距离边30px）
            self.speed_chart_label.move(
                window_width - self.speed_chart_label.width() - margin, margin
            )
            print(
                f"📍 速度趋势图位置: ({window_width - self.speed_chart_label.width() - margin}, {margin})"
            )

            # 3. 底部图表区域（左下角）
            # 计算图表区域的Y坐标，确保贴底边
            chart_area_y = window_height - chart_height - margin

            # 确保图表区域不超出窗口边界
            if chart_area_y < margin:
                chart_area_y = margin

            # 散点图（左侧）
            scatter_x = margin
            self.scatter_canvas.move(scatter_x, chart_area_y)
            print(f"📍 散点图位置: ({scatter_x}, {chart_area_y})")

            # 热力图（散点图右侧，间隔margin）
            heatmap_x = scatter_x + self.scatter_canvas.width() + margin
            self.heatmap_canvas.move(heatmap_x, chart_area_y)
            print(f"📍 热力图位置: ({heatmap_x}, {chart_area_y})")

            print(f"✅ UI位置更新完成")

        except Exception as e:
            print(f"❌ 更新UI位置失败: {e}")
            logger.error(f"Failed to update UI positions: {str(e)}")

    def _force_refresh_layout(self):
        """强制刷新布局"""
        try:
            print("🔄 强制刷新布局...")

            # 再次调用位置更新
            self._update_ui_positions()

            # 强制重绘
            self.main_widget.update()
            self.scatter_canvas.update()
            self.heatmap_canvas.update()

            # 确保组件可见
            self.scatter_canvas.raise_()
            self.heatmap_canvas.raise_()

            print("✅ 布局强制刷新完成")

        except Exception as e:
            print(f"❌ 强制刷新布局失败: {e}")
            logger.error(f"Failed to force refresh layout: {str(e)}")

    def load_accumulated_training_time(self):
        """从存档加载累积的训练时长"""
        try:
            if not self.save_folder_path:
                print("⚠️ 未指定存档路径，训练时长从0开始")
                return
                
            training_file = os.path.join(self.save_folder_path, "training_time.txt")
            
            if os.path.exists(training_file):
                with open(training_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content and content.isdigit():
                        self.total_training_time = int(content)
                        print(f"⏱️ 加载累积训练时长: {self.total_training_time}秒")
                    else:
                        print("⚠️ 训练时长文件格式错误，从0开始")
                        self.total_training_time = 0
            else:
                print("⏱️ 训练时长文件不存在，从0开始")
                self.total_training_time = 0
                
        except Exception as e:
            print(f"❌ 加载累积训练时长失败: {e}")
            logger.error(f"加载累积训练时长失败: {str(e)}")
            self.total_training_time = 0

    def load_accumulated_shot_count(self):
        """加载累积的板数数据，从speed_data文件读取"""
        try:
            if self.save_folder_path:
                speed_file = os.path.join(
                    self.save_folder_path, "speed_data", "speed_data.csv"
                )
            else:
                speed_file = os.path.join("speed_data", "speed_data.csv")

            if os.path.exists(speed_file):
                # 读取CSV文件，获取最大的板数
                max_shot_count = 0
                with open(speed_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if "shot_count" in row and row["shot_count"].isdigit():
                            shot_count = int(row["shot_count"])
                            max_shot_count = max(max_shot_count, shot_count)

                # 设置累积的板数
                if max_shot_count > 0:
                    self.trajectory_recorder.shot_count = max_shot_count
                    print(f"📊 加载累积板数: {max_shot_count}")
                else:
                    print("📊 未找到有效的板数数据")
            else:
                print("📊 速度数据文件不存在，板数从0开始")

        except Exception as e:
            print(f"❌ 加载累积板数失败: {e}")
            logger.error(f"Failed to load accumulated shot count: {str(e)}")

    def reset_playback_state(self):
        """重置播放状态但不重置累积数据"""
        try:
            # 暂停播放
            self.pause()

            # 重置播放状态
            self.trajectory_index = 0
            self.complete_trajectory = []

            # 重置3D可视化
            self.plt.pos_list = [
                np.full([3], None) for _ in range(self.plt.pos_list_memory_lenth)
            ]
            self.plt.n = 0
            self.plt.updatePlot()

            # 更新热力图、散点图和速度图表显示
            self.update_heatmap_display()
            self.update_scatter_display()
            self.update_speed_chart()

            print("🔄 播放状态已重置，累积数据保持不变")

        except Exception as e:
            print(f"❌ 重置播放状态失败: {e}")
            logger.error(f"Failed to reset playback state: {str(e)}")

    def update_table_size(self):
        """更新球台大小"""
        # 计算球台在窗口中的显示大小
        window_ratio = self.main_widget.width() / self.main_widget.height()
        table_ratio = 2740 / 1525  # 球台实际长宽比

        if window_ratio > table_ratio:
            # 如果窗口更宽，以高度为基准
            display_height = self.main_widget.height() * 0.8
            display_width = display_height * table_ratio
        else:
            # 如果窗口更窄，以宽度为基准
            display_width = self.main_widget.width() * 0.8
            display_height = display_width / table_ratio

        # 计算缩放比例
        self.scale_x = display_width / 2740
        self.scale_y = display_height / 1525

        # 计算球台在窗口中的位置（居中显示）
        self.offset_x = (self.main_widget.width() - display_width) / 2
        self.offset_y = (self.main_widget.height() - display_height) / 2

    def start_lcm_subscription(self):
        """启动LCM订阅，接收实时球位置数据"""
        if not LCM_AVAILABLE:
            print("❌ LCM库不可用，无法启动订阅")
            return

        try:
            # 创建LCM实例
            self.lcm_instance = lcm.LCM()

            # 订阅球位置数据通道
            self.lcm_subscription = self.lcm_instance.subscribe(
                "EXAMPLE", self._handle_lcm_message
            )

            # 启动LCM处理线程
            self.lcm_running = True
            self.lcm_thread = threading.Thread(target=self._lcm_worker, daemon=True)
            self.lcm_thread.start()

            # 启动LCM健康检查定时器
            self.lcm_health_timer = QTimer()
            self.lcm_health_timer.timeout.connect(self._lcm_health_check)
            self.lcm_health_timer.start(5000)  # 每5秒检查一次

            # 切换到实时渲染模式
            self.switch_to_real_time_mode()

            print("✅ LCM订阅已启动，正在监听EXAMPLE通道")
            logger.info("LCM订阅已启动，通道: EXAMPLE")

        except Exception as e:
            error_msg = f"启动LCM订阅失败: {str(e)}"
            print(f"❌ {error_msg}")
            logger.error(error_msg)
            QMessageBox.critical(
                self.main_widget,
                "LCM Error",
                f"Failed to start LCM subscription:\n{str(e)}",
            )

    # def _handle_lcm_message(self, channel, data):
    #     """处理LCM消息的回调函数"""
    #     try:
    #         # 检查是否处于实时模式
    #         if not hasattr(self, 'data_source') or self.data_source != "real_time":
    #             print("⚠️ 收到LCM消息但未处于实时模式，忽略消息")
    #             return

    #         # 验证数据完整性
    #         if not data or len(data) == 0:
    #             print("⚠️ 收到空的LCM消息数据")
    #             return

    #         # 解析消息
    #         try:
    #             msg = exlcm.ball_position_t.decode(data)
    #         except Exception as decode_error:
    #             # 如果标准解码失败，尝试兼容解码（忽略fingerprint检查）
    #             try:
    #                 if len(data) >= 40:  # 8字节fingerprint + 32字节数据
    #                     import struct
    #                     # 跳过fingerprint，直接解析数据部分
    #                     data_part = data[8:]
    #                     timestamp, x, y, z = struct.unpack('>qddd', data_part)
                        
    #                     # 手动创建消息对象
    #                     msg = exlcm.ball_position_t()
    #                     msg.timestamp = timestamp
    #                     msg.x = x
    #                     msg.y = y  
    #                     msg.z = z
                        
    #                     print(f"✅ 兼容解码成功: 时间戳={msg.timestamp}, X={msg.x:.3f}, Y={msg.y:.3f}, Z={msg.z:.3f}")
    #                 else:
    #                     raise ValueError(f"数据长度不足: {len(data)}")
    #             except Exception as compat_error:
    #                 print(f"❌ LCM消息解码失败: {decode_error}")
    #                 print(f"❌ 兼容解码也失败: {compat_error}")
    #                 print(f"🔍 原始数据长度: {len(data) if data else 0}")
    #                 if data and len(data) >= 8:
    #                     fingerprint = int.from_bytes(data[:8], 'big')
    #                     expected = exlcm.ball_position_t._get_hash_recursive([])
    #                     print(f"🔍 收到fingerprint: 0x{fingerprint:016x}")
    #                     print(f"🔍 期望fingerprint: 0x{expected:016x}")
    #                 logger.error(f"LCM消息解码失败: {str(decode_error)}")
    #                 return
            
    #         # 验证消息对象完整性
    #         if not hasattr(msg, 'x') or not hasattr(msg, 'y') or not hasattr(msg, 'z') or not hasattr(msg, 'timestamp'):
    #             print(f"❌ LCM消息格式不完整，缺少必要字段")
    #             print(f"🔍 消息对象属性: {dir(msg)}")
    #             return
            
    #         # 验证数据有效性
    #         if msg.x is None or msg.y is None or msg.z is None:
    #             print("⚠️ 收到无效的LCM消息，坐标包含None值")
    #             return
                
 
    #         # 使用当前时间的微秒时间戳，忽略原始时间戳
    #         import time
    #         timestamp_microseconds = int(time.time() * 1_000_000)
            
    #         # 保存当前原始时间戳用于下次比较（保持微秒精度）
    #         self._last_lcm_timestamp_microseconds = timestamp_microseconds

    #         # 直接调用process_realtime_position_update方法处理实时数据
    #         # 将微秒时间戳转换为秒，但仅在需要时进行转换
    #         timestamp_sec = timestamp_microseconds / 1000000.0
    #         try:
    #             self.process_realtime_position_update([msg.x, msg.y, msg.z], timestamp_sec)
    #         except Exception as process_error:
    #             print(f"❌ 处理实时位置更新失败: {process_error}")
    #             logger.error(f"处理实时位置更新失败: {str(process_error)}")
    #             # 不中断LCM消息处理流程

    #         print(
    #             f"📡 接收到实时数据: 时间={timestamp_microseconds}μs ({timestamp_sec:.6f}s), 位置=({msg.x:.3f}, {msg.y:.3f}, {msg.z:.3f})"
    #         )

    #     except Exception as e:
    #         error_msg = f"处理LCM消息失败: {str(e)}"
    #         print(f"❌ {error_msg}")
    #         logger.error(error_msg)
            
    #         # 记录详细的错误信息
    #         import traceback
    #         print(f"🔍 详细错误信息:")
    #         traceback.print_exc()
            
    #         # 尝试继续处理，不中断实时数据流
    #         # 如果错误持续发生，可能需要重建LCM实例

    def _handle_lcm_message(self, channel, data):
        """处理来自 LCM 的实时消息"""
        # 基础状态过滤
        if not hasattr(self, 'data_source') or self.data_source != "real_time":
            return

        try:
            # 1. 解码消息
            msg = exlcm.ball_position_t.decode(data)
            current_ts = time.time()

            # [乒乓球评估]调用处理器
            res = self.processor.process_realtime_step([msg.x, msg.y, msg.z], current_ts)
            filtered_pos, speed, events = res
            
            if filtered_pos is not None:
                # --- 新增：评估模式抓取数据 ---
                if self.is_evaluating_serve:
                    self.serve_data.append({'pos': filtered_pos, 'time': current_ts})
                    
                    # 如果检测到落点，自动停止并分析
                    if events.get("landing_detected"):
                        # 延迟一点点停止，为了抓取到撞击瞬间的完整轨迹
                        QTimer.singleShot(300, self.stop_serve_evaluation)
            # ---------------------------
            
            # 2. 调用处理器（执行滤波、去噪、落点分析等核心算法）
            res = self.processor.process_realtime_step([msg.x, msg.y, msg.z], current_ts)
            filtered_pos, speed, events = res
            
            # 如果是噪点被处理器拦截，则不进行渲染
            if filtered_pos is None: 
                return

            # 3. 更新 3D 渲染 (addNewBall 很快，但 updatePlot 很耗资源，因此控制刷新率)
            self.plt.addNewBall(filtered_pos)
            if events.get("frame_count", 0) % 2 == 0: # 隔帧刷新 OpenGL 提高流畅度
                self.plt.updatePlot()

            # 4. 更新 UI 文本显示
            self.update_speed_display(speed, events["shot_count"])

            # 5. 处理重大事件记录
            if events["y_trend_changed"]:
                # 记录速度数据
                self.processor.recorder.record_speed_data(current_ts, speed, filtered_pos, self.processor.prev_pos)
                self.update_speed_chart()
            
            if events["landing_detected"]:
                # 更新落点图表
                self.update_heatmap_display()
                self.update_scatter_display()

        except Exception as e:
            # 这里打印错误，方便你在优化算法时调试
            print(f"📡 LCM Process Error: {e}")


    def _lcm_worker(self):
        """LCM工作线程，持续处理消息"""
        consecutive_errors = 0  # 连续错误计数
        max_consecutive_errors = 5  # 最大连续错误次数
        reconnect_delay = 1.0  # 重连延迟（秒）
        
        try:
            print("🔄 LCM工作线程已启动")
            
            while self.lcm_running:
                try:
                    # 检查LCM实例是否有效
                    if not self.lcm_instance:
                        print("⚠️ LCM实例无效，尝试重新创建...")
                        self._recreate_lcm_instance()
                        continue
                    
                    # 使用线程锁保护LCM操作
                    with self.lcm_lock:
                        if self.lcm_operation_in_progress:
                            # 如果其他操作正在进行，等待一下
                            time.sleep(0.01)
                            continue
                            
                        self.lcm_operation_in_progress = True
                        
                        try:
                            # 处理LCM消息（非阻塞，超时100ms）
                            message_count = self.lcm_instance.handle_timeout(100)
                            
                            if message_count > 0:
                                # 有消息被处理，重置错误计数
                                consecutive_errors = 0
                                if consecutive_errors > 0:
                                    print(f"✅ LCM消息处理恢复正常，连续错误计数重置")
                            elif message_count < 0:
                                # 处理错误
                                consecutive_errors += 1
                                print(f"⚠️ LCM处理返回错误: {message_count}, 连续错误: {consecutive_errors}")
                                
                                if consecutive_errors >= max_consecutive_errors:
                                    print(f"❌ 连续错误过多，尝试重新创建LCM实例")
                                    self._recreate_lcm_instance()
                                    consecutive_errors = 0
                                    time.sleep(reconnect_delay)
                        finally:
                            self.lcm_operation_in_progress = False
                    
                    # 短暂休眠，避免CPU占用过高
                    time.sleep(0.01)
                    
                except Exception as e:
                    consecutive_errors += 1
                    error_msg = f"LCM工作线程循环异常: {str(e)}"
                    print(f"❌ {error_msg}")
                    logger.error(error_msg)
                    
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"❌ 连续异常过多，尝试重新创建LCM实例")
                        try:
                            self._recreate_lcm_instance()
                            consecutive_errors = 0
                        except Exception as reconnect_error:
                            print(f"❌ 重新创建LCM实例失败: {reconnect_error}")
                            logger.error(f"重新创建LCM实例失败: {str(reconnect_error)}")
                        
                        time.sleep(reconnect_delay)
                    else:
                        time.sleep(0.1)  # 短暂等待后继续

        except Exception as e:
            error_msg = f"LCM工作线程严重异常: {str(e)}"
            print(f"❌ {error_msg}")
            logger.error(error_msg)
            
            # 尝试自动恢复
            try:
                print("🔄 尝试自动恢复LCM连接...")
                self._recreate_lcm_instance()
            except Exception as recovery_error:
                print(f"❌ 自动恢复失败: {recovery_error}")
                logger.error(f"自动恢复失败: {str(recovery_error)}")
                
        finally:
            print("🔄 LCM工作线程已结束")
            # 如果线程意外退出但lcm_running仍为True，尝试重启
            if self.lcm_running:
                print("⚠️ LCM工作线程意外退出，尝试重启...")
                QTimer.singleShot(2000, self._restart_lcm_worker)  # 2秒后尝试重启

  

    def stop_lcm_subscription(self):
        """停止LCM订阅"""
        try:
            print("🔄 正在停止LCM订阅...")
            
            # 停止健康检查定时器
            if hasattr(self, 'lcm_health_timer') and self.lcm_health_timer:
                self.lcm_health_timer.stop()
                self.lcm_health_timer = None
                print("✅ LCM健康检查定时器已停止")

            # 等待一小段时间确保定时器完全停止
            time.sleep(0.1)

            # 停止工作线程
            self.lcm_running = False

            if self.lcm_thread and self.lcm_thread.is_alive():
                print("🔄 等待LCM工作线程结束...")
                self.lcm_thread.join(timeout=2.0)
                if self.lcm_thread.is_alive():
                    print("⚠️ LCM工作线程未能在2秒内结束")

            # 等待线程锁释放
            if hasattr(self, 'lcm_lock'):
                with self.lcm_lock:
                    pass  # 确保锁被释放

            # 清理订阅和实例
            if self.lcm_subscription and self.lcm_instance:
                try:
                    self.lcm_instance.unsubscribe(self.lcm_subscription)
                    print("✅ LCM订阅已取消")
                except Exception as e:
                    print(f"⚠️ 取消LCM订阅失败: {e}")
                self.lcm_subscription = None

            if self.lcm_instance:
                try:
                    # LCM对象没有close方法，只需要取消订阅即可
                    print("✅ LCM实例已清理")
                except Exception as e:
                    print(f"⚠️ 清理LCM实例失败: {e}")
                self.lcm_instance = None

            # 重置相关状态
            self.lcm_thread = None
            self._lcm_health_error_count = 0 if hasattr(self, '_lcm_health_error_count') else 0
            self.lcm_operation_in_progress = False

            print("✅ LCM订阅已完全停止")
            logger.info("LCM订阅已完全停止")

        except Exception as e:
            print(f"❌ 停止LCM订阅失败: {str(e)}")
            logger.error(f"停止LCM订阅失败: {str(e)}")

    def switch_to_real_time_mode(self):
        """切换到实时渲染模式"""
        try:
            # 停止加载轨迹渲染
            if self.is_rendering:
                self.pause()
                print("⏸️ 已停止加载轨迹渲染")

            # 重置加载轨迹相关状态
            self.trajectory_index = 0
            self.complete_trajectory = []

            # 清空3D视图中的轨迹
            if hasattr(self, "plt") and self.plt:
                try:
                    # 安全地清空3D视图数据
                    if hasattr(self.plt, 'pos_list_memory_lenth'):
                        self.plt.pos_list = [
                            np.full([3], None) for _ in range(self.plt.pos_list_memory_lenth)
                        ]
                        self.plt.n = 0
                        self.plt.updatePlot()
                        print("✅ 3D视图轨迹已清空")
                    else:
                        print("⚠️ 3D视图缺少pos_list_memory_lenth属性")
                except Exception as e:
                    print(f"⚠️ 清空3D视图失败: {e}")
                    logger.error(f"清空3D视图失败: {str(e)}")

            # 初始化实时模式相关变量
            self._realtime_trajectory_index = 0
            self.prev_realtime_pos = None
            self.prev_realtime_time = None
            self.prev_realtime_y_trend = None  # 添加Y轴趋势跟踪
            self.current_time = time.time()
            self.frame_count = 0
            
            # 重置落点分析状态
            self.landing_analyzer.reset_landing_analysis()
            
            # 确保轨迹记录模块处于正确状态
            if hasattr(self, 'trajectory_recorder'):
                # 加载累积的拍数数据
                self.load_accumulated_shot_count()
                print(f"📊 实时模式初始化完成，当前拍数: {self.trajectory_recorder.get_shot_count()}")
                
                # 更新显示，确保一致性
                shot_count = self.trajectory_recorder.get_shot_count()
                self.update_speed_display(0.0, shot_count)
                self.update_speed_chart()

            self.processor.reset() # 确保处理器也清空了历史滤波状态
            # 更新数据源标识
            self.data_source = "real_time"
            
            # 清空缓冲区
            self.raw_data_buffer.clear()
            self.last_valid_pos = None

            print("✅ 已切换到实时渲染模式")

        except Exception as e:
            print(f"❌ 切换到实时渲染模式失败: {str(e)}")
            logger.error(f"切换到实时渲染模式失败: {str(e)}")

    def switch_to_trajectory_mode(self):
        """切换到加载轨迹渲染模式"""
        try:
            # 停止实时渲染
            if self.lcm_running:
                self.stop_lcm_subscription()
                print("⏸️ 已停止实时渲染")
            
            # 更新实时渲染按钮状态
            if hasattr(self, 'realtime_render_btn'):
                self.realtime_render_btn.setChecked(False)
                self.realtime_render_btn.setText("Real-time Render")
                print("🔄 实时渲染按钮状态已重置")

            # 清空实时数据列表
            self.real_time_positions = []
            self.real_time_timestamps = []
            if hasattr(self, "real_time_pos_list"):
                self.real_time_pos_list = []

            # 清空3D视图中的实时轨迹
            if hasattr(self, "plt") and self.plt:
                self.plt.pos_list = [
                    np.full([3], None) for _ in range(self.plt.pos_list_memory_lenth)
                ]
                self.plt.n = 0
                self.plt.updatePlot()

            # 更新数据源标识
            self.data_source = "local_trajectory"

            print("✅ 已切换到加载轨迹渲染模式")

        except Exception as e:
            print(f"❌ 切换到加载轨迹渲染模式失败: {str(e)}")
            logger.error(f"切换到加载轨迹渲染模式失败: {str(e)}")

    def cleanup(self):
        """清理资源"""
        try:
            # 停止加载轨迹渲染
            if self.is_rendering:
                self.pause()
                print("⏹️ 加载轨迹渲染已停止")

            # 停止LCM订阅
            self.stop_lcm_subscription()

            # 停止采集程序 - 使用简洁的endprocess方式
            self._force_kill_collection_process()

            # 清理3D视图
            if hasattr(self, "plt") and self.plt:
                self.plt.pos_list = [
                    np.full([3], None) for _ in range(self.plt.pos_list_memory_lenth)
                ]
                self.plt.n = 0
                self.plt.updatePlot()
                print("🧹 3D视图已清理")

            print("✅ 资源清理完成")

        except Exception as e:
            print(f"❌ 资源清理失败: {str(e)}")
            logger.error(f"资源清理失败: {str(e)}")

    def get_realtime_status(self):
        """获取实时模式的当前状态信息"""
        try:
            status = {
                "data_source": getattr(self, 'data_source', 'unknown'),
                "lcm_running": getattr(self, 'lcm_running', False),
                "frame_count": getattr(self, 'frame_count', 0),
                "current_time": getattr(self, 'current_time', 0),
                "realtime_trajectory_index": getattr(self, '_realtime_trajectory_index', 0),
                "prev_realtime_pos": getattr(self, 'prev_realtime_pos', None),
                "prev_realtime_time": getattr(self, 'prev_realtime_time', None),
                "prev_realtime_y_trend": getattr(self, 'prev_realtime_y_trend', None),
                "shot_count": self.trajectory_recorder.get_shot_count() if hasattr(self, 'trajectory_recorder') else 0,
                "rally_count": self.trajectory_recorder.get_rally_count() if hasattr(self, 'trajectory_recorder') else 0
            }
            return status
        except Exception as e:
            print(f"❌ 获取实时状态失败: {e}")
            return {"error": str(e)}

    def print_realtime_status(self):
        """打印实时模式的当前状态信息"""
        try:
            status = self.get_realtime_status()
            print("\n" + "="*50)
            print("📊 实时模式状态信息")
            print("="*50)
            for key, value in status.items():
                print(f"{key}: {value}")
            print("="*50)
        except Exception as e:
            print(f"❌ 打印实时状态失败: {e}")



    def _recreate_lcm_instance(self):
        """重新创建LCM实例和订阅"""
        try:
            print("🔄 重新创建LCM实例...")
            
            # 清理旧的实例
            if self.lcm_subscription and self.lcm_instance:
                try:
                    self.lcm_instance.unsubscribe(self.lcm_subscription)
                    print("✅ 旧订阅已取消")
                except Exception as e:
                    print(f"⚠️ 取消旧订阅失败: {e}")
                self.lcm_subscription = None
            
            if self.lcm_instance:
                try:
                    # LCM对象没有close方法，只需要取消订阅即可
                    # 让Python垃圾回收器处理LCM实例
                    print("✅ 旧LCM实例已清理")
                except Exception as e:
                    print(f"⚠️ 清理旧LCM实例失败: {e}")
                self.lcm_instance = None
            
            # 创建新的LCM实例
            self.lcm_instance = lcm.LCM()
            
            # 重新订阅
            self.lcm_subscription = self.lcm_instance.subscribe(
                "EXAMPLE", self._handle_lcm_message
            )
            
            print("✅ LCM实例重建成功")
            logger.info("LCM实例重建成功")
            
        except Exception as e:
            error_msg = f"重建LCM实例失败: {str(e)}"
            print(f"❌ {error_msg}")
            logger.error(error_msg)
            raise

    def _restart_lcm_worker(self):
        """重启LCM工作线程"""
        try:
            if not self.lcm_running:
                print("⚠️ LCM已停止运行，不重启工作线程")
                return
                
            if self.lcm_thread and self.lcm_thread.is_alive():
                print("⚠️ LCM工作线程仍在运行，不重启")
                return
            
            print("🔄 重启LCM工作线程...")
            
            # 启动新的工作线程
            self.lcm_thread = threading.Thread(target=self._lcm_worker, daemon=True)
            self.lcm_thread.start()
            
            print("✅ LCM工作线程重启成功")
            logger.info("LCM工作线程重启成功")
            
        except Exception as e:
            error_msg = f"重启LCM工作线程失败: {str(e)}"
            print(f"❌ {error_msg}")
            logger.error(error_msg)

    def _check_lcm_health(self):
        """检查LCM连接健康状态"""
        try:
            if not self.lcm_running:
                return False
                
            if not self.lcm_thread or not self.lcm_thread.is_alive():
                print("⚠️ LCM工作线程已死亡")
                return False
                
            if not self.lcm_instance:
                print("⚠️ LCM实例无效")
                return False
                
            # 尝试简单的LCM操作来测试连接
            try:
                # 非阻塞检查，超时10ms
                self.lcm_instance.handle_timeout(10)
                return True
            except Exception as e:
                print(f"⚠️ LCM健康检查失败: {e}")
                return False
                
        except Exception as e:
            print(f"❌ LCM健康检查异常: {e}")
            return False

    def _lcm_health_check(self):
        """定期检查LCM连接健康状态"""
        try:
            if not self.lcm_running:
                return
                
            # 检查工作线程状态
            if not self.lcm_thread or not self.lcm_thread.is_alive():
                print("⚠️ LCM健康检查: 工作线程已死亡，尝试重启...")
                self._restart_lcm_worker()
                return
                
            # 检查LCM实例状态
            if not self.lcm_instance:
                print("⚠️ LCM健康检查: 实例无效，尝试重建...")
                self._recreate_lcm_instance()
                return
                
            # 使用线程锁保护LCM操作，避免并发调用
            with self.lcm_lock:
                if self.lcm_operation_in_progress:
                    # 如果工作线程正在操作，跳过这次检查
                    return
                    
                # 检查消息处理状态（通过简单的超时操作）
                try:
                    # 非阻塞检查，超时10ms
                    result = self.lcm_instance.handle_timeout(10)
                    if result < 0:
                        print(f"⚠️ LCM健康检查: 处理返回错误 {result}")
                    # 即使有错误也不立即重建，给系统一些恢复时间
                        
                except Exception as e:
                    print(f"⚠️ LCM健康检查: 实例操作异常 {e}")
                    # 如果连续出现异常，考虑重建实例
                    if not hasattr(self, '_lcm_health_error_count'):
                        self._lcm_health_error_count = 0
                    self._lcm_health_error_count += 1
                    
                    if self._lcm_health_error_count >= 3:
                        print("❌ LCM健康检查: 连续错误过多，重建实例...")
                        self._recreate_lcm_instance()
                        self._lcm_health_error_count = 0
                        
        except Exception as e:
            print(f"❌ LCM健康检查异常: {e}")
            logger.error(f"LCM健康检查异常: {str(e)}")

    def safe_shutdown(self):
        """安全的程序关闭，确保资源按正确顺序清理"""
        try:
            print("🔄 开始安全关闭程序...")
            
            # 1. 首先停止所有定时器
            if hasattr(self, 'training_timer') and self.training_timer:
                self.training_timer.stop()
                print("✅ 训练定时器已停止")
                
            if hasattr(self, 'lcm_health_timer') and self.lcm_health_timer:
                self.lcm_health_timer.stop()
                print("✅ LCM健康检查定时器已停止")
            
            # 2. 保存当前训练时长到存档
            try:
                total_seconds = self.calculate_training_time()
                self.save_training_time_to_archive(total_seconds)
                print(f"💾 最终训练时长已保存: {total_seconds:.0f}秒")
            except Exception as e:
                print(f"⚠️ 保存最终训练时长失败: {e}")
            
            # 3. 停止LCM订阅
            if hasattr(self, 'lcm_running') and self.lcm_running:
                self.stop_lcm_subscription()
            
            # 4. 关闭采集程序进程 - 使用简洁的endprocess方式
            self._force_kill_collection_process()
            
            # 5. 等待一小段时间确保所有线程完全停止
            time.sleep(0.2)
            
            # 6. 关闭数据记录
            if hasattr(self, 'trajectory_recorder'):
                try:
                    self.trajectory_recorder.close_speed_data_recording()
                    print("✅ 速度数据记录已关闭")
                except Exception as e:
                    print(f"⚠️ 关闭速度数据记录失败: {e}")
                    
            if hasattr(self, 'landing_analyzer'):
                try:
                    self.landing_analyzer.close_landing_data_recording()
                    print("✅ 落点数据记录已关闭")
                except Exception as e:
                    print(f"⚠️ 关闭落点数据记录失败: {e}")
            
            # 6. 最后关闭3D视图（OpenGL上下文）
            if hasattr(self, 'plt') and self.plt:
                try:
                    # 清空3D视图数据，避免OpenGL错误
                    self.plt.pos_list = []
                    self.plt.n = 0
                    print("✅ 3D视图数据已清空")
                except Exception as e:
                    print(f"⚠️ 清空3D视图失败: {e}")
            
            print("✅ 程序安全关闭完成")
            
        except Exception as e:
            print(f"❌ 程序安全关闭失败: {e}")
            logger.error(f"程序安全关闭失败: {str(e)}")

    def handle_close_event(self, event):
        """处理窗口关闭事件"""
        try:
            print("🔄 窗口关闭事件触发...")
            
            # 执行安全关闭
            self.safe_shutdown()
            
            # 调用回调函数（如果存在）
            if self.on_close_callback:
                try:
                    self.on_close_callback()
                except Exception as e:
                    print(f"⚠️ 关闭回调执行失败: {e}")
            
            # 接受关闭事件
            event.accept()
            
        except Exception as e:
            print(f"❌ 处理关闭事件失败: {e}")
            # 即使失败也要接受关闭事件
            event.accept()

    def save_training_time_to_archive(self, total_seconds):
        """保存训练时长到存档文件"""
        try:
            if not self.save_folder_path:
                return
                
            training_file = os.path.join(self.save_folder_path, "training_time.txt")
            
            # 确保目录存在
            os.makedirs(os.path.dirname(training_file), exist_ok=True)
            
            # 保存训练时长（秒）
            with open(training_file, "w", encoding="utf-8") as f:
                f.write(str(int(total_seconds)))
            
            print(f"💾 训练时长已保存到存档: {total_seconds:.0f}秒")
            
        except Exception as e:
            print(f"❌ 保存训练时长失败: {e}")
            logger.error(f"保存训练时长失败: {str(e)}")

    def validate_and_reset_training_time(self):
        """验证并重置异常的训练时长值"""
        try:
            current_total = self.calculate_training_time()
            
            # 检查训练时长是否异常（超过24小时）
            if current_total > 86400:  # 24小时 = 86400秒
                print(f"⚠️ 训练时长异常: {current_total:.0f}秒 (>24小时)，重置为0")
                self.total_training_time = 0
                self.training_start_time = time.time()
                return True
                
            # 检查是否为负数
            if current_total < 0:
                print(f"⚠️ 训练时长为负数: {current_total:.0f}秒，重置为0")
                self.total_training_time = 0
                self.training_start_time = time.time()
                return True
                
            # 检查是否为NaN或Inf
            if np.isnan(current_total) or np.isinf(current_total):
                print(f"⚠️ 训练时长无效: {current_total}，重置为0")
                self.total_training_time = 0
                self.training_start_time = time.time()
                return True
                
            return False
            
        except Exception as e:
            print(f"❌ 验证训练时长失败: {e}")
            # 出错时强制重置
            self.total_training_time = 0
            self.training_start_time = time.time()
            return True

    def get_formatted_training_time(self):
        """获取格式化的训练时长字符串"""
        try:
            total_seconds = self.calculate_training_time()
            
            # 验证时长值
            if self.validate_and_reset_training_time():
                total_seconds = 0
            
            # 转换为时:分:秒格式
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = int(total_seconds % 60)
            
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
        except Exception as e:
            print(f"❌ 格式化训练时长失败: {e}")
            return "00:00:00"
        
    # [新增] 发球评估相关方法
    def toggle_serve_evaluation(self):
        """切换发球评估模式"""
        if self.eval_serve_btn.isChecked():
            self.start_serve_evaluation()
        else:
            self.stop_serve_evaluation()

    def start_serve_evaluation(self):
        """开始评估：清空数据，等待发球"""
        self.is_evaluating_serve = True
        self.serve_data = []
        self.eval_serve_btn.setText("Waiting...")
        print("🎾 进入发球评估模式：等待发球...")

    def stop_serve_evaluation(self):
        """停止评估：恢复按钮，进行分析"""
        self.is_evaluating_serve = False
        self.eval_serve_btn.setChecked(False)
        self.eval_serve_btn.setText("Evaluate Serve")
        
        if len(self.serve_data) > 5: # 至少要有几个点才分析
            self.analyze_serve_quality()
        else:
            print("❌ 未记录到有效的发球数据")

    # def analyze_serve_quality(self):
    #     """计算并显示发球质量报告"""
    #     try:
    #         start_point = self.serve_data[0]['pos']
    #         end_point = self.serve_data[-1]['pos']
    #         start_time = self.serve_data[0]['time']
    #         end_time = self.serve_data[-1]['time']
    #         duration = end_time - start_time
            
    #         # 计算总飞行距离（累加每一帧的距离）
    #         total_dist = 0
    #         for i in range(1, len(self.serve_data)):
    #             p1 = np.array(self.serve_data[i-1]['pos'])
    #             p2 = np.array(self.serve_data[i]['pos'])
    #             total_dist += np.linalg.norm(p2 - p1)
            
    #         # 计算平均速度 (mm/s -> m/s)
    #         avg_speed = (total_dist / 1000.0) / duration if duration > 0 else 0
            
    #         # 生成评语
    #         quality = "普通"
    #         if avg_speed > 12.0: quality = "极快"
    #         elif avg_speed < 4.0: quality = "过慢"
            
    #         # 简单的落点判断 (假设 x=0 是中线)
    #         landing_x = end_point[0]
    #         if abs(landing_x) > 600:
    #             quality += " (大角度)"
            
    #         msg = (f"⏱️ 飞行时间: {duration:.2f} s\n"
    #                f"🚀 平均球速: {avg_speed:.2f} m/s\n"
    #                f"📍 落点坐标: X={end_point[0]:.0f}, Y={end_point[1]:.0f}\n"
    #                f"⭐ 综合评价: {quality}")
                   
    #         print(f"\n📊 === 发球评估报告 ===\n{msg}")
            
    #         QMessageBox.information(self.main_widget, "Serve Analysis", msg)
            
    #     except Exception as e:
    #         print(f"❌ 分析发球数据失败: {e}")

    def analyze_serve_quality(self):
        """计算质量并弹出精美的评估报告"""
        report = self.processor.get_serve_features(self.serve_data)
        
        if not report:
            QMessageBox.warning(self.main_widget, "提醒", "采集点过少，无法分析发球。")
            return

        # 保存到历史记录
        self.save_serve_to_history(report)

        # 构建展示信息
        result_text = (
            f"📊 <b style='color:#E67E22;'>发球评测报告</b><br><br>"
            f"🚀 <b>最高瞬时球速:</b> {report['max_speed']:.2f} m/s<br>"
            f"🔝 <b>轨迹最高点:</b> {report['peak_height']:.1f} mm<br>"
            f"📍 <b>落点坐标:</b> ({report['landing_x']:.0f}, {report['landing_y']:.0f})<br>"
            f"⏱️ <b>飞行时长:</b> {report['duration']:.2f} s<br><br>"
            f"💡 <i>提示：发球数据已自动归档，可点击历史统计查看。</i>"
        )
        
        msg_box = QMessageBox(self.main_widget)
        msg_box.setWindowTitle("发球诊断完成")
        msg_box.setText(result_text)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()
        
        # 分析完后立即更新热力图（查看历史分布）
        self.update_heatmap_display()

def main():
    """主函数."""
    try:
        app = QtWidgets.QApplication([])

        # 在创建主窗口之前设置应用程序图标
        try:
            icon_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logo.jpg"
            )
            if os.path.exists(icon_path):
                try:
                    import cv2
                    from PyQt5.QtGui import QIcon, QImage, QPixmap

                    # 使用OpenCV读取图片并转换为QPixmap
                    img = cv2.imread(icon_path)
                    if img is not None:
                        # 转换BGR到RGB
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        height, width, channel = img_rgb.shape
                        bytes_per_line = 3 * width

                        # 创建QPixmap
                        qimg = QImage(
                            img_rgb.data,
                            width,
                            height,
                            bytes_per_line,
                            QImage.Format_RGB888,
                        )
                        pixmap = QPixmap.fromImage(qimg)

                        # 创建图标并设置到应用程序
                        icon = QIcon(pixmap)
                        app.setWindowIcon(icon)
                        print(f"✅ 应用程序图标已设置: {icon_path}")
                    else:
                        print(f"⚠️ 无法读取图片文件: {icon_path}")
                except ImportError:
                    print(f"⚠️ cv2模块未安装，跳过图标设置")
                except Exception as e:
                    print(f"⚠️ 设置应用程序图标失败: {e}")
            else:
                print(f"⚠️ 图标文件不存在: {icon_path}")
        except Exception as e:
            print(f"⚠️ 设置应用程序图标失败: {e}")

        # 创建默认存档路径（当直接运行模拟器时）
        default_save_folder = os.path.join(
            os.path.dirname(__file__), "saves", "default"
        )
        if not os.path.exists(default_save_folder):
            os.makedirs(default_save_folder)

        simulator = BallTrajectorySimulator(default_save_folder, None)
        app.exec_()
    except Exception as e:
        logger.error(f"Program failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
