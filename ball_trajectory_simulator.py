import os
import sys
import time
import threading
import numpy as np
import subprocess
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFrame, QMessageBox, QFileDialog)

# 导入自定义组件
from .chart_renderer import ChartRenderer
from .plot3D_230704 import plot3D
from .trajectory_processor import TrajectoryProcessor  # 导入新拆分的处理器

# LCM 导入逻辑 (保持原样)
try:
    import lcm
    import exlcm
    LCM_AVAILABLE = True
except ImportError:
    LCM_AVAILABLE = False

# 导入之前定义的各种自定义 Button 类 (FuturisticButton, RecordButton, 等...)
# [此处省略重复的 Button 样式代码，建议保留在文件顶部]

class BallTrajectorySimulator:
    def __init__(self, save_folder_path=None, on_close_callback=None):
        self.save_folder_path = save_folder_path
        self.on_close_callback = on_close_callback
        
        # --- 核心：初始化轨迹处理器 ---
        self.processor = TrajectoryProcessor(save_folder_path)
        self.chart_renderer = ChartRenderer(save_folder_path)
        
        # UI 与 渲染 状态
        self.is_rendering = False
        self.is_recording = False
        self.data_source = None
        self.lcm_running = False
        
        # 初始化 3D 视图
        self._setup_3d_view()
        # 初始化 UI 界面
        self._init_main_ui()

    def _setup_3d_view(self):
        # 球台角点加载逻辑 (保持原样)
        corners = np.array([[-1370, -762.5, 0], [1370, -762.5, 0], [1370, 762.5, 0], [-1370, 762.5, 0]])
        self.plt = plot3D((1200, 800), corners, None, None, True, 5)

    def _handle_lcm_message(self, channel, data):
        """处理来自 LCM 的实时消息"""
        # 基础状态过滤
        if not hasattr(self, 'data_source') or self.data_source != "real_time":
            return

        try:
            # 1. 解码消息
            msg = exlcm.ball_position_t.decode(data)
            current_ts = time.time()
            
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


    # --- UI 事件处理 ---
    def start_local_monitor(self):
        """启动本地采集程序并切换到实时模式"""
        # [省略启动进程的代码，保留原逻辑]
        self.switch_to_real_time_mode()
        if LCM_AVAILABLE: self.start_lcm_subscription()

    def switch_to_real_time_mode(self):
        """切换状态并重置处理器"""
        self.processor.reset() # 确保处理器也清空了历史滤波状态
        self.data_source = "real_time"
        self.processor.reset()
        if hasattr(self, "plt"):
            self.plt.pos_list = [np.full([3], None) for _ in range(self.plt.pos_list_memory_lenth)]
            self.plt.n = 0
            self.plt.updatePlot()
        print("✅ 已切换到实时处理器模式")

    def update_speed_display(self, speed, shot_count):
        # 更新 self.speed_label 文本内容 (保持原样)
        pass

    # [此处保留所有的 UI 辅助函数: update_speed_chart, draw_heatmap_plot, resizeEvent, cleanup 等]

def main():
    app = QtWidgets.QApplication(sys.argv)
    # 创建存档路径
    save_path = os.path.join(os.path.dirname(__file__), "saves", "default")
    os.makedirs(save_path, exist_ok=True)
    
    sim = BallTrajectorySimulator(save_path)
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()