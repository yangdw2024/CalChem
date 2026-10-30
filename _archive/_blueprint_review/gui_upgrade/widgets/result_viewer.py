"""
结果展示组件 — 多标签页展示计算结果。

标签页:
  1. 能量分解   — 总能量及各分量 (SCF / DFT / Post-HF)
  2. 分子轨道   — 轨道能量、占据数 (标记 HOMO)
  3. 轨道能级图  — 可视化轨道能级，标注HOMO/LUMO
  4. 分子性质   — 偶极矩等
  5. 频率/振动  — 振动频率与热力学量，支持振动动画
  6. 激发态     — 激发能、振子强度

单位换算:
  轨道能量: Eh × 27.2114 = eV
  偶极矩:   a.u. × 2.541746 = Debye

方法:
  display_results(results: dict) — 填充所有标签页
  clear()                        — 清空所有标签页
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional

import numpy as np

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QLabel, QPushButton, QSlider, QGroupBox,
    QSplitter,
)

try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ── 物理常数 ──
HARTREE_TO_EV: float = 27.2114
AU_TO_DEBYE: float = 2.541746
NM_PER_EV: float = 1239.841984  # λ(nm) = 1239.84 / E(eV)


class OrbitalLevelPlot(QWidget):
    """分子轨道能级图绘制组件。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._mo_energies: List[float] = []
        self._mo_occupations: List[float] = []
        self._homo_index: int = -1
        self._lumo_index: int = -1
        self._selected_orbital: int = -1

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # 提示标签
        self.lbl_info = QLabel("点击轨道查看详细信息")
        layout.addWidget(self.lbl_info)

        if MATPLOTLIB_AVAILABLE:
            # Matplotlib 图形
            self.figure = Figure(figsize=(5, 6))
            self.canvas = FigureCanvas(self.figure)
            self.ax = self.figure.add_subplot(111)
            layout.addWidget(self.canvas, 1)

            # 连接点击事件
            self.canvas.mpl_connect('button_press_event', self._on_click)

            # 详情显示区域
            self.lbl_detail = QLabel("")
            self.lbl_detail.setStyleSheet(
                "QLabel { padding: 6px; background: #f0f0f0; "
                "border: 1px solid #ccc; }")
            layout.addWidget(self.lbl_detail)
        else:
            # 如果没有matplotlib，显示提示
            self.lbl_placeholder = QLabel(
                "需要安装 matplotlib 才能显示轨道能级图\n"
                "请运行: pip install matplotlib")
            self.lbl_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.lbl_placeholder.setStyleSheet(
                "QLabel { color: #666; padding: 20px; }")
            layout.addWidget(self.lbl_placeholder, 1)

    def set_orbital_data(
        self,
        energies: List[float],
        occupations: List[float],
        homo_index: int = -1,
        lumo_index: int = -1
    ) -> None:
        """设置轨道数据并绘制能级图。"""
        self._mo_energies = energies
        self._mo_occupations = occupations
        self._homo_index = homo_index
        self._lumo_index = lumo_index

        # 自动计算HOMO/LUMO索引（如果未提供）
        if homo_index < 0:
            for i, occ in enumerate(occupations):
                if occ > 0:
                    self._homo_index = i

        if lumo_index < 0:
            self._lumo_index = self._homo_index + 1

        self._update_plot()

    def _update_plot(self) -> None:
        """更新能级图。"""
        if not MATPLOTLIB_AVAILABLE:
            return

        self.ax.clear()

        if not self._mo_energies:
            self.ax.text(0.5, 0.5, "无轨道数据",
                        ha='center', va='center', fontsize=12)
            self.canvas.draw()
            return

        n_orbitals = len(self._mo_energies)
        energies_ev = [e * HARTREE_TO_EV for e in self._mo_energies]

        # 绘制能级线
        for i, (e, occ) in enumerate(zip(energies_ev, self._mo_occupations)):
            # 轨道颜色：占据轨道蓝色，空轨道红色
            if occ > 0:
                color = '#2244AA'
                marker = 'o'
                markersize = 8
            else:
                color = '#AA2244'
                marker = 'o'
                markersize = 6

            # 绘制能级线
            self.ax.plot([0.3, 0.7], [e, e], color=color, linewidth=2)

            # 标注HOMO/LUMO
            if i == self._homo_index:
                self.ax.text(0.75, e, 'HOMO', fontsize=10, va='center',
                            color='#2244AA', fontweight='bold')
                self.ax.plot(0.5, e, marker='s', color='#2244AA',
                            markersize=10, zorder=5)
            elif i == self._lumo_index and self._lumo_index < n_orbitals:
                self.ax.text(0.75, e, 'LUMO', fontsize=10, va='center',
                            color='#AA2244', fontweight='bold')
                self.ax.plot(0.5, e, marker='s', color='#AA2244',
                            markersize=10, zorder=5)

            # 显示轨道序号
            self.ax.text(0.25, e, str(i + 1), fontsize=8, va='center',
                        ha='right', color='#444')

        # 设置坐标轴
        self.ax.set_xlim(0, 1.0)
        self.ax.set_ylabel('轨道能量 (eV)', fontsize=11)
        self.ax.set_xlabel('', fontsize=11)
        self.ax.set_title('分子轨道能级图', fontsize=12, fontweight='bold')

        # 去除x轴刻度
        self.ax.set_xticks([])

        # 添加网格
        self.ax.grid(True, axis='y', linestyle='--', alpha=0.5)

        self.figure.tight_layout()
        self.canvas.draw()

        # 更新信息标签
        self.lbl_info.setText(
            f"轨道总数: {n_orbitals}  |  "
            f"占据轨道: {sum(1 for o in self._mo_occupations if o > 0)}  |  "
            f"HOMO-LUMO能隙: {energies_ev[self._lumo_index] - energies_ev[self._homo_index]:.3f} eV")

    def _on_click(self, event) -> None:
        """处理点击事件，显示轨道详情。"""
        if not MATPLOTLIB_AVAILABLE or not self._mo_energies:
            return

        if event.inaxes != self.ax:
            return

        # 查找最近的轨道
        click_y = event.ydata
        energies_ev = [e * HARTREE_TO_EV for e in self._mo_energies]

        min_dist = float('inf')
        nearest_idx = -1

        for i, e in enumerate(energies_ev):
            dist = abs(click_y - e)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        if nearest_idx >= 0 and min_dist < 0.5:  # 容差范围
            self._selected_orbital = nearest_idx
            self._show_orbital_detail(nearest_idx)

    def _show_orbital_detail(self, idx: int) -> None:
        """显示轨道详细信息。"""
        if idx >= len(self._mo_energies):
            return

        e_hartree = self._mo_energies[idx]
        e_ev = e_hartree * HARTREE_TO_EV
        occ = self._mo_occupations[idx] if idx < len(self._mo_occupations) else 0.0

        orbital_type = "占据轨道" if occ > 0 else "空轨道"

        if idx == self._homo_index:
            orbital_type += " (HOMO)"
        elif idx == self._lumo_index:
            orbital_type += " (LUMO)"

        self.lbl_detail.setText(
            f"轨道 #{idx + 1}: {orbital_type}\n"
            f"能量: {e_hartree:.6f} Hartree = {e_ev:.4f} eV\n"
            f"占据数: {occ:.3f}")

    def clear(self) -> None:
        """清空数据。"""
        self._mo_energies = []
        self._mo_occupations = []
        self._homo_index = -1
        self._lumo_index = -1
        self._selected_orbital = -1

        if MATPLOTLIB_AVAILABLE:
            self.ax.clear()
            self.ax.text(0.5, 0.5, "无数据", ha='center', va='center')
            self.canvas.draw()

        self.lbl_info.setText("点击轨道查看详细信息")
        self.lbl_detail.setText("")


class VibrationAnimator(QWidget):
    """振动模式动画播放组件。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._symbols: List[str] = []
        self._base_coords: np.ndarray = np.zeros((0, 3))
        self._mode_vectors: List[np.ndarray] = []
        self._frequencies: List[float] = []
        self._current_mode: int = -1
        self._animating: bool = False
        self._animation_frame: int = 0
        self._amplitude: float = 0.5  # 振动幅度系数

        # 信号回调 - 用于更新3D视图
        self._coords_callback: Optional[callable] = None

        self._init_ui()

        # 动画定时器
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._animate_frame)

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # 控制面板
        control_group = QGroupBox("振动动画控制")
        control_layout = QHBoxLayout(control_group)

        self.lbl_freq = QLabel("选择振动模式")
        control_layout.addWidget(self.lbl_freq)

        self.btn_play = QPushButton("播放")
        self.btn_play.clicked.connect(self._toggle_animation)
        control_layout.addWidget(self.btn_play)

        self.btn_stop = QPushButton("停止")
        self.btn_stop.clicked.connect(self._stop_animation)
        control_layout.addWidget(self.btn_stop)

        # 幅度滑块
        control_layout.addWidget(QLabel("幅度:"))
        self.slider_amplitude = QSlider(Qt.Orientation.Horizontal)
        self.slider_amplitude.setMinimum(1)
        self.slider_amplitude.setMaximum(100)
        self.slider_amplitude.setValue(50)
        self.slider_amplitude.valueChanged.connect(self._on_amplitude_changed)
        control_layout.addWidget(self.slider_amplitude)

        layout.addWidget(control_group)

        # 提示标签
        self.lbl_info = QLabel(
            "提示: 点击频率列表中的频率可以播放对应的振动模式")
        self.lbl_info.setStyleSheet("QLabel { color: #666; padding: 4px; }")
        layout.addWidget(self.lbl_info)

        # 状态显示
        self.lbl_status = QLabel("状态: 未播放")
        self.lbl_status.setStyleSheet(
            "QLabel { padding: 6px; background: #f0f0f0; }")
        layout.addWidget(self.lbl_status)

    def set_vibration_data(
        self,
        symbols: List[str],
        base_coords: np.ndarray,
        mode_vectors: List[np.ndarray],
        frequencies: List[float],
        coords_callback: Optional[callable] = None
    ) -> None:
        """设置振动数据。

        Args:
            symbols: 原子符号列表
            base_coords: 基础坐标 (Å)
            mode_vectors: 振动模式向量列表 (每个为 n_atoms x 3)
            frequencies: 振动频率列表 (cm⁻¹)
            coords_callback: 用于更新分子视图的回调函数
        """
        self._symbols = symbols
        self._base_coords = np.array(base_coords, dtype=float)
        self._mode_vectors = [np.array(v, dtype=float) for v in mode_vectors]
        self._frequencies = frequencies
        self._coords_callback = coords_callback

        self._current_mode = -1
        self._stop_animation()

        self.lbl_info.setText(
            f"共有 {len(frequencies)} 个振动模式")

    def set_mode(self, mode_idx: int) -> None:
        """设置当前振动模式。"""
        if mode_idx < 0 or mode_idx >= len(self._frequencies):
            return

        self._current_mode = mode_idx
        freq = self._frequencies[mode_idx]

        self.lbl_freq.setText(
            f"当前模式 #{mode_idx + 1}: {freq:.2f} cm⁻¹")

        # 重置动画帧
        self._animation_frame = 0

        # 更新状态
        self.lbl_status.setText(
            f"状态: 已选择模式 #{mode_idx + 1}")

    def _toggle_animation(self) -> None:
        """切换动画播放状态。"""
        if self._animating:
            self._stop_animation()
        else:
            self._start_animation()

    def _start_animation(self) -> None:
        """开始动画。"""
        if self._current_mode < 0:
            self.lbl_status.setText("状态: 请先选择振动模式")
            return

        if len(self._base_coords) == 0:
            self.lbl_status.setText("状态: 无坐标数据")
            return

        self._animating = True
        self._animation_frame = 0
        self.btn_play.setText("暂停")

        # 根据频率设置动画速度（60帧周期）
        freq = self._frequencies[self._current_mode]
        # 周期时间 = 1 / freq (cm⁻¹转换为Hz需要换算，但简化处理)
        # 这里使用固定帧率，每帧间隔根据频率调整
        interval = max(20, min(100, int(1000 / (freq / 100))))
        self._timer.start(interval)

        self.lbl_status.setText(
            f"状态: 播放模式 #{self._current_mode + 1}")

    def _stop_animation(self) -> None:
        """停止动画。"""
        self._animating = False
        self._timer.stop()
        self.btn_play.setText("播放")

        # 恢复到基础坐标
        if self._coords_callback and len(self._base_coords) > 0:
            self._coords_callback(self._symbols, self._base_coords)

        self.lbl_status.setText("状态: 已停止")

    def _animate_frame(self) -> None:
        """执行一帧动画。"""
        if not self._animating or self._current_mode < 0:
            return

        if self._current_mode >= len(self._mode_vectors):
            return

        # 简谐振动: coords(t) = base + amplitude * sin(2πt) * mode_vector
        self._animation_frame += 1
        phase = 2 * np.pi * (self._animation_frame / 60.0)  # 60帧一个周期
        displacement = self._amplitude * np.sin(phase)

        mode_vec = self._mode_vectors[self._current_mode]
        # 归一化模式向量（避免过大位移）
        norm = np.linalg.norm(mode_vec)
        if norm > 0:
            mode_vec = mode_vec / norm * 0.5  # 最大位移 0.5 Å

        new_coords = self._base_coords + displacement * mode_vec

        # 调用回调更新视图
        if self._coords_callback:
            self._coords_callback(self._symbols, new_coords)

    def _on_amplitude_changed(self, value: int) -> None:
        """调整振动幅度。"""
        self._amplitude = value / 50.0  # 0.02 ~ 2.0

    def clear(self) -> None:
        """清空数据。"""
        self._symbols = []
        self._base_coords = np.zeros((0, 3))
        self._mode_vectors = []
        self._frequencies = []
        self._current_mode = -1

        self._stop_animation()
        self.lbl_freq.setText("选择振动模式")
        self.lbl_info.setText("提示: 点击频率列表中的频率可以播放对应的振动模式")
        self.lbl_status.setText("状态: 未播放")


class ResultViewer(QWidget):
    """结果展示与分析组件 (多标签页)。"""

    # 方法分类 (与 calchem_bridge 保持一致)
    DFT_METHODS: set = {"LDA", "PBE", "B3LYP", "TPSS", "BLYP", "PBE0",
                        "PW91", "revPBE", "RPBE", "SCAN", "M06L", "B2PLYP"}
    POST_HF_METHODS: set = {"MP2", "MP4", "CCSD", "CCSD(T)", "CCD", "CASSCF"}

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._calc_time: Optional[float] = None
        self._current_coords: np.ndarray = np.zeros((0, 3))
        self._current_symbols: List[str] = []
        self._init_ui()

    # ── UI 构建 ──────────────────────────────────────────
    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # 计算耗时显示
        self.lbl_time = QLabel("")
        self.lbl_time.setStyleSheet(
            "QLabel { padding: 4px; background: #e8e8e8; "
            "border: 1px solid #ccc; font-weight: bold; }")
        layout.addWidget(self.lbl_time)

        self.tabs = QTabWidget(self)
        layout.addWidget(self.tabs, 1)

        # 1. 能量分解
        self.energy_table = self._make_table(
            ["项目", "能量 (Hartree)", "能量 (eV)"])
        self.tabs.addTab(self.energy_table, "能量分解")

        # 2. 分子轨道 (表格)
        self.mo_table = self._make_table(
            ["#", "轨道能量 (Eh)", "轨道能量 (eV)", "占据数", "备注"])
        self.tabs.addTab(self.mo_table, "分子轨道")

        # 3. 【新增】轨道能级图
        self.orbital_plot = OrbitalLevelPlot()
        self.tabs.addTab(self.orbital_plot, "轨道能级图")

        # 4. 分子性质
        self.prop_table = self._make_table(["性质", "数值", "单位"])
        self.tabs.addTab(self.prop_table, "分子性质")

        # 5. 【增强】频率/振动 - 使用分割器包含表格和动画控制
        freq_widget = QWidget()
        freq_layout = QVBoxLayout(freq_widget)
        freq_layout.setContentsMargins(2, 2, 2, 2)

        # 频率表格
        self.freq_table = self._make_table(
            ["#", "频率 (cm⁻¹)", "强度", "对称性"])
        freq_layout.addWidget(self.freq_table)

        # 振动动画控制器
        self.vib_animator = VibrationAnimator()
        freq_layout.addWidget(self.vib_animator)

        # 连接频率表格点击事件
        self.freq_table.cellClicked.connect(self._on_freq_clicked)

        self.tabs.addTab(freq_widget, "频率/振动")

        # 6. 激发态
        self.exc_table = self._make_table(
            ["#", "激发能 (eV)", "波长 (nm)", "振子强度"])
        self.tabs.addTab(self.exc_table, "激发态")

    def _make_table(self, headers: List[str]) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setAlternatingRowColors(True)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setVisible(False)
        return table

    # ── 辅助方法 ────────────────────────────────────────
    @staticmethod
    def _num_item(value: float, fmt: str = "{:.6f}") -> QTableWidgetItem:
        """创建右对齐的数值单元格。"""
        item = QTableWidgetItem(fmt.format(float(value)))
        item.setTextAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return item

    @staticmethod
    def _text_item(text: str, align_right: bool = False) -> QTableWidgetItem:
        """创建文本单元格。"""
        item = QTableWidgetItem(text)
        flag = (Qt.AlignmentFlag.AlignRight if align_right
                else Qt.AlignmentFlag.AlignLeft)
        item.setTextAlignment(flag | Qt.AlignmentFlag.AlignVCenter)
        return item

    @staticmethod
    def _as_list(arr: Any) -> List[Any]:
        """将 ndarray / tuple / 标量统一转为 list。"""
        if arr is None:
            return []
        if isinstance(arr, np.ndarray):
            return arr.tolist()
        if isinstance(arr, (list, tuple)):
            return list(arr)
        return [arr]

    def _set_rows(self, table: QTableWidget,
                  rows: List[List[QTableWidgetItem]]) -> None:
        """批量填充表格行。"""
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, item in enumerate(row):
                if c < table.columnCount():
                    table.setItem(r, c, item)

    # ── 公共 API ────────────────────────────────────────
    def clear(self) -> None:
        """清空所有标签页内容。"""
        self.lbl_time.setText("")
        for i in range(self.tabs.count()):
            w = self.tabs.widget(i)
            if isinstance(w, QTableWidget):
                w.setRowCount(0)
            elif isinstance(w, OrbitalLevelPlot):
                w.clear()
            elif isinstance(w, VibrationAnimator):
                w.clear()

        self._calc_time = None

    def set_coords_callback(self, callback: callable) -> None:
        """设置坐标更新回调函数（用于振动动画）。"""
        self.vib_animator._coords_callback = callback

    def display_results(
        self,
        results: Dict[str, Any],
        calc_time: Optional[float] = None
    ) -> None:
        """填充所有标签页。空字典仅清空。

        Args:
            results: 计算结果字典
            calc_time: 计算耗时（秒）
        """
        self.clear()
        if not results:
            return

        # 存储结果数据（用于频率点击事件）
        self._stored_results = results

        self._calc_time = calc_time
        if calc_time is not None:
            self.lbl_time.setText(f"计算耗时: {calc_time:.2f} 秒")

        # 保存当前坐标（用于振动动画）
        if "symbols" in results and "coordinates" in results:
            self._current_symbols = list(results["symbols"])
            self._current_coords = np.array(results["coordinates"], dtype=float)

        self._fill_energy(results)
        self._fill_molecular_orbitals(results)
        self._fill_properties(results)
        self._fill_frequencies(results)
        self._fill_excited_states(results)

    # ── 1. 能量分解 ──
    def _fill_energy(self, results: Dict[str, Any]) -> None:
        method = results.get("method", "RHF")
        rows: List[List[QTableWidgetItem]] = []

        # ── 通用 SCF 分量 ──
        e_1e = results.get("e_1e")
        e_2e = results.get("e_2e")
        e_nn = results.get("e_nn")
        e_tot = results.get("e_tot")

        if e_1e is not None:
            rows.append([self._text_item("单电子能量 (E_1e)"),
                         self._num_item(e_1e),
                         self._num_item(e_1e * HARTREE_TO_EV)])
        if e_2e is not None:
            rows.append([self._text_item("双电子能量 (E_2e)"),
                         self._num_item(e_2e),
                         self._num_item(e_2e * HARTREE_TO_EV)])
        if e_nn is not None:
            rows.append([self._text_item("核排斥能 (E_nn)"),
                         self._num_item(e_nn),
                         self._num_item(e_nn * HARTREE_TO_EV)])

        # ── DFT 分量 ──
        if method in self.DFT_METHODS or "e_xc" in results:
            e_xc = results.get("e_xc")
            if e_xc is not None:
                rows.append([self._text_item("交换相关能 (E_xc)"),
                             self._num_item(e_xc),
                             self._num_item(e_xc * HARTREE_TO_EV)])
            e_d3 = results.get("e_d3")
            if e_d3 is not None:
                rows.append([self._text_item("D3 色散校正 (E_D3)"),
                             self._num_item(e_d3),
                             self._num_item(e_d3 * HARTREE_TO_EV)])

        # ── Post-HF 分量 ──
        if method in self.POST_HF_METHODS:
            e_corr = results.get("e_corr")
            e_tot_ph = results.get("e_tot_post_hf")

            if e_corr is not None:
                rows.append([self._text_item("相关能 (E_corr)"),
                             self._num_item(e_corr),
                             self._num_item(e_corr * HARTREE_TO_EV)])

            # MP4 各阶贡献
            if method == "MP4":
                for key, label in [
                    ("e_mp2", "MP2 相关能"),
                    ("e_mp3", "MP3 相关能"),
                    ("e_mp4_s", "MP4(S) 贡献"),
                    ("e_mp4_d", "MP4(D) 贡献"),
                    ("e_mp4_t", "MP4(T) 贡献"),
                    ("e_mp4_q", "MP4(Q) 贡献"),
                ]:
                    v = results.get(key)
                    if v is not None:
                        rows.append([self._text_item(label),
                                     self._num_item(v),
                                     self._num_item(v * HARTREE_TO_EV)])

            # CCSD(T) 分量
            if method == "CCSD(T)":
                e_ccsd = results.get("e_corr_ccsd")
                e_pt = results.get("e_corr_pt")
                if e_ccsd is not None:
                    rows.append([self._text_item("CCSD 相关能"),
                                 self._num_item(e_ccsd),
                                 self._num_item(e_ccsd * HARTREE_TO_EV)])
                if e_pt is not None:
                    rows.append([self._text_item("(T) 微扰校正"),
                                 self._num_item(e_pt),
                                 self._num_item(e_pt * HARTREE_TO_EV)])

            # CASSCF CI 本征值
            if method == "CASSCF":
                ci_vals = results.get("ci_eigenvalues")
                if ci_vals is not None:
                    ci_list = self._as_list(ci_vals)
                    for i, v in enumerate(ci_list):
                        rows.append([self._text_item(f"CI 根 {i + 1}"),
                                     self._num_item(v),
                                     self._num_item(v * HARTREE_TO_EV)])

            if e_tot_ph is not None:
                rows.append([self._text_item(f"{method} 总能量"),
                             self._num_item(e_tot_ph),
                             self._num_item(e_tot_ph * HARTREE_TO_EV)])

        # ── SCF 总能量 ──
        if e_tot is not None:
            rows.append([self._text_item("SCF 总能量 (E_tot)"),
                         self._num_item(e_tot),
                         self._num_item(e_tot * HARTREE_TO_EV)])

        # ── 收敛信息 ──
        n_cycles = results.get("n_cycles")
        if n_cycles is not None:
            rows.append([self._text_item("SCF 迭代步数"),
                         self._text_item(str(n_cycles), align_right=True),
                         self._text_item("步", align_right=True)])
        converged = results.get("converged")
        if converged is not None:
            rows.append([self._text_item("收敛状态"),
                         self._text_item("已收敛" if converged else "未收敛",
                                         align_right=True),
                         self._text_item("")])

        # ── 计算耗时 ──
        if self._calc_time is not None:
            rows.append([self._text_item("计算耗时"),
                         self._text_item(f"{self._calc_time:.2f}",
                                         align_right=True),
                         self._text_item("秒", align_right=True)])

        self._set_rows(self.energy_table, rows)

    # ── 2. 分子轨道 ──
    def _normalize_mo(
        self, mo_energy: Any, mo_occ: Any
    ) -> List[Tuple[List[float], List[float], str]]:
        """归一化为 [(energies, occs, label), ...]。

        - RHF/ROHF: 单组, label=""
        - UHF/UKS:  alpha + beta 双组, label="α"/"β"
        """
        result: List[Tuple[List[float], List[float], str]] = []
        if mo_energy is None:
            return result

        # tuple -> (alpha, beta) 双组
        if isinstance(mo_energy, tuple) and len(mo_energy) == 2:
            e_a = [float(x) for x in self._as_list(mo_energy[0])]
            e_b = [float(x) for x in self._as_list(mo_energy[1])]
            o_a: List[float] = []
            o_b: List[float] = []
            if mo_occ is not None and isinstance(mo_occ, tuple) \
                    and len(mo_occ) == 2:
                o_a = [float(x) for x in self._as_list(mo_occ[0])]
                o_b = [float(x) for x in self._as_list(mo_occ[1])]
            result.append((e_a, o_a, "α"))
            result.append((e_b, o_b, "β"))
        else:
            e = [float(x) for x in self._as_list(mo_energy)]
            o = [float(x) for x in self._as_list(mo_occ)] \
                if mo_occ is not None else []
            result.append((e, o, ""))
        return result

    def _find_homo_lumo(
        self, energies: List[float], occs: List[float]
    ) -> Tuple[int, int]:
        """查找 HOMO 和 LUMO 索引。"""
        homo_idx = -1
        for i, occ in enumerate(occs):
            if occ > 0:
                homo_idx = i

        lumo_idx = homo_idx + 1 if homo_idx >= 0 else 0

        return homo_idx, lumo_idx

    def _build_mo_rows(
        self, groups: List[Tuple[List[float], List[float], str]]
    ) -> List[List[QTableWidgetItem]]:
        """构建分子轨道表格行, 标记 HOMO (带 *)。"""
        rows: List[List[QTableWidgetItem]] = []
        idx = 1
        for energies, occs, label in groups:
            n = len(energies)
            # HOMO = 最后一个占据数 > 0 的轨道
            homo_idx = -1
            for i in range(len(occs)):
                if occs[i] > 0:
                    homo_idx = i

            for i in range(n):
                e = energies[i]
                occ = occs[i] if i < len(occs) else 0.0
                note = label
                if i == homo_idx:
                    note = (note + " " if note else "") + "HOMO*"
                rows.append([
                    self._text_item(str(idx), align_right=True),
                    self._num_item(e, "{:.6f}"),
                    self._num_item(e * HARTREE_TO_EV, "{:.4f}"),
                    self._num_item(occ, "{:.3f}"),
                    self._text_item(note),
                ])
                idx += 1
        return rows

    def _fill_molecular_orbitals(self, results: Dict[str, Any]) -> None:
        # 优先使用 bridge 拆分后的 alpha/beta 字段
        mo_energy: Any = results.get("mo_energy")
        if "mo_energy_alpha" in results or "mo_energy_beta" in results:
            e_a = results.get("mo_energy_alpha")
            e_b = results.get("mo_energy_beta")
            mo_energy = (e_a, e_b) if e_b is not None else e_a
        mo_occ = results.get("mo_occ")

        groups = self._normalize_mo(mo_energy, mo_occ)
        if not groups:
            return

        # 填充表格
        rows = self._build_mo_rows(groups)
        self._set_rows(self.mo_table, rows)

        # 【新增】填充轨道能级图
        # 对于 UHF，只显示 alpha 轨道
        all_energies = []
        all_occs = []

        # 只取第一组（RHF或UHF alpha轨道）
        energies, occs, label = groups[0]
        all_energies = list(energies)
        all_occs = list(occs)

        homo_idx, lumo_idx = self._find_homo_lumo(all_energies, all_occs)
        self.orbital_plot.set_orbital_data(all_energies, all_occs, homo_idx, lumo_idx)

    # ── 3. 分子性质 ──
    def _fill_properties(self, results: Dict[str, Any]) -> None:
        rows: List[List[QTableWidgetItem]] = []

        dipole = results.get("dipole")
        if dipole is not None:
            dip = self._as_list(dipole)
            if len(dip) >= 3:
                dx, dy, dz = float(dip[0]), float(dip[1]), float(dip[2])
                dmag = float(np.sqrt(dx * dx + dy * dy + dz * dz))
                rows.append([self._text_item("偶极矩 |μ|"),
                             self._num_item(dmag, "{:.6f}"),
                             self._text_item("a.u.", align_right=True)])
                rows.append([self._text_item("偶极矩 |μ|"),
                             self._num_item(dmag * AU_TO_DEBYE, "{:.6f}"),
                             self._text_item("Debye", align_right=True)])
                rows.append([self._text_item("偶极矩 μ_x"),
                             self._num_item(dx, "{:.6f}"),
                             self._text_item("a.u.", align_right=True)])
                rows.append([self._text_item("偶极矩 μ_y"),
                             self._num_item(dy, "{:.6f}"),
                             self._text_item("a.u.", align_right=True)])
                rows.append([self._text_item("偶极矩 μ_z"),
                             self._num_item(dz, "{:.6f}"),
                             self._text_item("a.u.", align_right=True)])

        if not rows:
            rows.append([self._text_item("(无分子性质数据)"),
                         self._text_item(""), self._text_item("")])
        self._set_rows(self.prop_table, rows)

    # ── 4. 频率/振动 ──
    def _fill_frequencies(self, results: Dict[str, Any]) -> None:
        rows: List[List[QTableWidgetItem]] = []
        freqs = results.get("frequencies")
        if freqs is None:
            return

        freq_list = self._as_list(freqs)
        thermo = results.get("thermochemistry", {})

        # 获取振动模式向量（如果有）
        mode_vectors = results.get("mode_vectors", [])
        if isinstance(mode_vectors, np.ndarray):
            mode_vectors = mode_vectors.tolist()

        # 设置振动动画数据
        if len(self._current_coords) > 0 and len(self._current_symbols) > 0:
            self.vib_animator.set_vibration_data(
                self._current_symbols,
                self._current_coords,
                mode_vectors if mode_vectors else [],
                freq_list
            )

        for i, f in enumerate(freq_list):
            rows.append([
                self._text_item(str(i + 1), align_right=True),
                self._num_item(float(f), "{:.2f}"),
                self._text_item("--"),
                self._text_item(""),
            ])

        # 热力学量追加
        if isinstance(thermo, dict) and thermo:
            rows.append([self._text_item(""), self._text_item(""),
                         self._text_item(""), self._text_item("")])
            rows.append([self._text_item("— 热力学量 —"),
                         self._text_item(""), self._text_item(""),
                         self._text_item("")])
            for key, label in [
                ("zpe", "零点能 (ZPE)"),
                ("u_thermo", "内能 (U)"),
                ("h_thermo", "焓 (H)"),
                ("g_thermo", "自由能 (G)"),
                ("cv", "等容热容 (Cv)"),
                ("entropy", "熵 (S)"),
            ]:
                v = thermo.get(key)
                if v is not None:
                    rows.append([
                        self._text_item(label),
                        self._num_item(float(v), "{:.6f}"),
                        self._text_item("a.u.", align_right=True),
                        self._text_item(""),
                    ])

        if not rows:
            rows.append([self._text_item("(无频率数据)"),
                         self._text_item(""), self._text_item(""),
                         self._text_item("")])
        self._set_rows(self.freq_table, rows)

    def _on_freq_clicked(self, row: int, col: int) -> None:
        """处理频率表格点击，设置振动模式。"""
        # 获取频率索引（排除热力学行）
        freqs = self._as_list(self.project_results.get("frequencies", []))

        if row < len(freqs):
            self.vib_animator.set_mode(row)

    @property
    def project_results(self) -> Dict[str, Any]:
        """临时存储结果数据（用于频率点击）。"""
        return getattr(self, '_stored_results', {})

    def _fill_excited_states(self, results: Dict[str, Any]) -> None:
        rows: List[List[QTableWidgetItem]] = []
        exc_energies = results.get("excitation_energies")
        osc_strengths = results.get("oscillator_strengths")
        if exc_energies is None:
            return

        e_list = self._as_list(exc_energies)
        o_list = self._as_list(osc_strengths) \
            if osc_strengths is not None else []

        # 单位判定: 若所有值 < 1.0, 视为 Hartree; 否则视为 eV
        raw_vals = [float(x) for x in e_list]
        is_hartree = all(abs(v) < 1.0 for v in raw_vals) if raw_vals else False

        for i, raw in enumerate(raw_vals):
            e_eV = raw * HARTREE_TO_EV if is_hartree else raw
            e_nm = NM_PER_EV / e_eV if e_eV > 1e-10 else 0.0
            osc = float(o_list[i]) if i < len(o_list) else 0.0
            rows.append([
                self._text_item(str(i + 1), align_right=True),
                self._num_item(e_eV, "{:.4f}"),
                self._num_item(e_nm, "{:.2f}"),
                self._num_item(osc, "{:.4f}"),
            ])

        if not rows:
            rows.append([self._text_item("(无激发态数据)"),
                         self._text_item(""), self._text_item(""),
                         self._text_item("")])
        self._set_rows(self.exc_table, rows)