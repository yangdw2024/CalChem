#!/usr/bin/env python3
"""
CalChem 主窗口 — 整合所有 GUI 组件。

布局: 三栏 QSplitter (左:分子编辑 / 中:3D视图+收敛曲线 / 右:参数+结果)
      + 底部作业监控 + 状态栏
"""
from __future__ import annotations

import os
import time
from collections import Counter
from typing import Optional

import numpy as np

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QSplitter,
    QTabWidget, QTableWidget, QTableWidgetItem, QFileDialog,
    QMessageBox, QLabel, QHeaderView, QAbstractItemView,
    QStyle,
)

from widgets.molecule_viewer import MoleculeViewer
from widgets.molecule_editor import MoleculeEditor
from widgets.parameter_panel import ParameterPanel
from widgets.result_viewer import ResultViewer
from widgets.job_monitor import JobMonitor
from widgets.convergence_plot import ConvergencePlot
from core.project import ProjectModel
from core.worker import CalcWorker


# 常见元素原子序数 (用于原子列表显示)
_ATOMIC_NUMBERS = {
    "H": 1, "HE": 2, "LI": 3, "BE": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "NE": 10, "NA": 11, "MG": 12, "AL": 13, "SI": 14, "P": 15,
    "S": 16, "CL": 17, "AR": 18, "K": 19, "CA": 20, "BR": 35, "I": 53,
}

# 分子片段库 - 标准几何坐标 (单位: Å)
_FRAGMENTS = {
    "水 (H2O)": {
        "symbols": ["O", "H", "H"],
        "coords": [
            [0.0000, 0.0000, 0.1173],  # O
            [0.0000, 0.7572, -0.4692],  # H
            [0.0000, -0.7572, -0.4692],  # H
        ],
    },
    "甲烷 (CH4)": {
        "symbols": ["C", "H", "H", "H", "H"],
        "coords": [
            [0.0000, 0.0000, 0.0000],  # C
            [0.6291, 0.6291, 0.6291],  # H
            [-0.6291, -0.6291, 0.6291],  # H
            [-0.6291, 0.6291, -0.6291],  # H
            [0.6291, -0.6291, -0.6291],  # H
        ],
    },
    "苯 (C6H6)": {
        "symbols": ["C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "H"],
        "coords": [
            [1.3970, 0.0000, 0.0000],  # C1
            [0.6985, 1.2100, 0.0000],  # C2
            [-0.6985, 1.2100, 0.0000],  # C3
            [-1.3970, 0.0000, 0.0000],  # C4
            [-0.6985, -1.2100, 0.0000],  # C5
            [0.6985, -1.2100, 0.0000],  # C6
            [2.4810, 0.0000, 0.0000],  # H1
            [1.2405, 2.1480, 0.0000],  # H2
            [-1.2405, 2.1480, 0.0000],  # H3
            [-2.4810, 0.0000, 0.0000],  # H4
            [-1.2405, -2.1480, 0.0000],  # H5
            [1.2405, -2.1480, 0.0000],  # H6
        ],
    },
    "乙烯 (C2H4)": {
        "symbols": ["C", "C", "H", "H", "H", "H"],
        "coords": [
            [0.0000, 0.0000, 0.6695],  # C1
            [0.0000, 0.0000, -0.6695],  # C2
            [0.0000, 0.9230, 1.2320],  # H1
            [0.0000, -0.9230, 1.2320],  # H2
            [0.0000, 0.9230, -1.2320],  # H3
            [0.0000, -0.9230, -1.2320],  # H4
        ],
    },
    "乙炔 (C2H2)": {
        "symbols": ["C", "C", "H", "H"],
        "coords": [
            [0.0000, 0.0000, 0.6015],  # C1
            [0.0000, 0.0000, -0.6015],  # C2
            [0.0000, 0.0000, 1.6635],  # H1
            [0.0000, 0.0000, -1.6635],  # H2
        ],
    },
    "氨 (NH3)": {
        "symbols": ["N", "H", "H", "H"],
        "coords": [
            [0.0000, 0.0000, 0.1173],  # N
            [0.0000, 0.9400, -0.2735],  # H1
            [0.8140, -0.4700, -0.2735],  # H2
            [-0.8140, -0.4700, -0.2735],  # H3
        ],
    },
    "二氧化碳 (CO2)": {
        "symbols": ["C", "O", "O"],
        "coords": [
            [0.0000, 0.0000, 0.0000],  # C
            [0.0000, 0.0000, 1.1620],  # O1
            [0.0000, 0.0000, -1.1620],  # O2
        ],
    },
}


class MainWindow(QMainWindow):
    """CalChem 主窗口。"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CalChem - 量子化学计算平台")
        self.resize(1400, 900)

        # 数据模型与工作线程
        self.project = ProjectModel()
        self.worker: Optional[CalcWorker] = None
        self._cycle_count = 0

        # 计算耗时记录
        self._calc_start_time: Optional[float] = None
        self._calc_elapsed_time: Optional[float] = None

        # 构建 UI
        self._init_ui()
        self._init_menu()
        self._init_toolbar()
        self._init_statusbar()
        self._connect_signals()

        # 内存刷新定时器
        self._mem_timer = QTimer(self)
        self._mem_timer.timeout.connect(self._update_memory)
        self._mem_timer.start(3000)
        self._update_memory()

        self._update_status()

    # ════════════════════════════════════════════════════
    #  UI 构建
    # ════════════════════════════════════════════════════

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # 主垂直分割: 上(三栏) | 下(作业监控, 可拖拽折叠)
        self.main_splitter = QSplitter(Qt.Orientation.Vertical)

        # 上: 三栏水平分割
        self.h_splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── 左栏: QTabWidget ──
        self.left_tabs = QTabWidget()
        self.editor = MoleculeEditor()
        self.left_tabs.addTab(self.editor, "分子编辑器")
        self.left_tabs.addTab(self._build_atom_list_widget(), "原子列表")

        # ── 中栏: 垂直分割 (3D 视图 + 收敛曲线) ──
        self.v_splitter = QSplitter(Qt.Orientation.Vertical)
        self.viewer = MoleculeViewer()
        self.conv_plot = ConvergencePlot()
        self.v_splitter.addWidget(self.viewer)
        self.v_splitter.addWidget(self.conv_plot)
        self.v_splitter.setSizes([450, 250])

        # ── 右栏: QTabWidget ──
        self.right_tabs = QTabWidget()
        self.param_panel = ParameterPanel()
        self.result_viewer = ResultViewer()
        self.right_tabs.addTab(self.param_panel, "参数面板")
        self.right_tabs.addTab(self.result_viewer, "结果展示")

        self.h_splitter.addWidget(self.left_tabs)
        self.h_splitter.addWidget(self.v_splitter)
        self.h_splitter.addWidget(self.right_tabs)
        self.h_splitter.setSizes([300, 650, 350])

        # ── 下: 作业监控 ──
        self.job_monitor = JobMonitor()

        self.main_splitter.addWidget(self.h_splitter)
        self.main_splitter.addWidget(self.job_monitor)
        self.main_splitter.setSizes([650, 200])

        layout.addWidget(self.main_splitter)

    def _build_atom_list_widget(self) -> QWidget:
        """构建只读原子列表面板 (Tab2)。"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(2, 2, 2, 2)

        self.lbl_formula = QLabel("分子式: (空)")
        self.lbl_formula.setStyleSheet(
            "QLabel { padding: 4px; background: #f5f5f5; "
            "border: 1px solid #ddd; font-weight: bold; }")
        layout.addWidget(self.lbl_formula)

        self.atom_list_table = QTableWidget(0, 6)
        self.atom_list_table.setHorizontalHeaderLabels(
            ["序号", "元素", "原子序数", "X (Å)", "Y (Å)", "Z (Å)"])
        self.atom_list_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self.atom_list_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.atom_list_table.setAlternatingRowColors(True)
        header = self.atom_list_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.atom_list_table.verticalHeader().setVisible(False)
        layout.addWidget(self.atom_list_table)

        return widget

    # ── 菜单栏 ──────────────────────────────────────────

    def _init_menu(self):
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu("文件(&F)")
        self.act_open = QAction("打开文件...", self)
        self.act_open.setShortcut(QKeySequence("Ctrl+O"))
        self.act_open.triggered.connect(self._open_file)
        file_menu.addAction(self.act_open)

        self.act_save = QAction("保存输入文件...", self)
        self.act_save.setShortcut(QKeySequence("Ctrl+S"))
        self.act_save.triggered.connect(self._save_gjf)
        file_menu.addAction(self.act_save)

        self.act_export = QAction("导出报告...", self)
        self.act_export.triggered.connect(self._export_report)
        file_menu.addAction(self.act_export)

        file_menu.addSeparator()

        self.act_exit = QAction("退出", self)
        self.act_exit.setShortcut(QKeySequence("Ctrl+Q"))
        self.act_exit.triggered.connect(self.close)
        file_menu.addAction(self.act_exit)

        # 编辑菜单
        edit_menu = menubar.addMenu("编辑(&E)")
        self.act_add_atom = QAction("添加原子", self)
        self.act_add_atom.setShortcut(QKeySequence("Insert"))
        self.act_add_atom.triggered.connect(self._add_atom)
        edit_menu.addAction(self.act_add_atom)

        self.act_del_atom = QAction("删除原子", self)
        self.act_del_atom.setShortcut(QKeySequence("Delete"))
        self.act_del_atom.triggered.connect(self._delete_atom)
        edit_menu.addAction(self.act_del_atom)

        edit_menu.addSeparator()

        self.act_clear_mol = QAction("清空分子", self)
        self.act_clear_mol.triggered.connect(self._clear_molecule)
        edit_menu.addAction(self.act_clear_mol)

        self.act_center = QAction("居中分子", self)
        self.act_center.triggered.connect(self._center_molecule)
        edit_menu.addAction(self.act_center)

        # 【新增】插入片段菜单
        edit_menu.addSeparator()
        fragment_menu = edit_menu.addMenu("插入片段")

        for frag_name in sorted(_FRAGMENTS.keys()):
            action = QAction(frag_name, self)
            action.triggered.connect(lambda checked, name=frag_name: self._insert_fragment(name))
            fragment_menu.addAction(action)

        # 计算菜单
        calc_menu = menubar.addMenu("计算(&C)")
        self.act_submit = QAction("提交计算", self)
        self.act_submit.setShortcut(QKeySequence("F5"))
        self.act_submit.triggered.connect(self._submit_calc)
        calc_menu.addAction(self.act_submit)

        self.act_cancel = QAction("终止计算", self)
        self.act_cancel.setShortcut(QKeySequence("Esc"))
        self.act_cancel.triggered.connect(self._cancel_calc)
        calc_menu.addAction(self.act_cancel)

        calc_menu.addSeparator()

        self.act_clear_results = QAction("清空结果", self)
        self.act_clear_results.triggered.connect(self._clear_results)
        calc_menu.addAction(self.act_clear_results)

        # 视图菜单
        view_menu = menubar.addMenu("视图(&V)")
        self.act_scale = QAction("比例模型", self)
        self.act_scale.setCheckable(True)
        self.act_scale.triggered.connect(self._on_toggle_style)
        view_menu.addAction(self.act_scale)

        self.act_axes = QAction("显示坐标轴", self)
        self.act_axes.setCheckable(True)
        self.act_axes.triggered.connect(self._on_toggle_axes)
        view_menu.addAction(self.act_axes)

        self.act_log_plot = QAction("对数收敛图", self)
        self.act_log_plot.setCheckable(True)
        self.act_log_plot.triggered.connect(self._on_toggle_log_plot)
        view_menu.addAction(self.act_log_plot)

        view_menu.addSeparator()

        self.act_toggle_monitor = QAction("显示作业监控", self)
        self.act_toggle_monitor.setCheckable(True)
        self.act_toggle_monitor.setChecked(True)
        self.act_toggle_monitor.triggered.connect(self._on_toggle_monitor)
        view_menu.addAction(self.act_toggle_monitor)

        # 帮助菜单
        help_menu = menubar.addMenu("帮助(&H)")
        self.act_about = QAction("关于", self)
        self.act_about.triggered.connect(self._about)
        help_menu.addAction(self.act_about)

    # ── 工具栏 ──────────────────────────────────────────

    def _init_toolbar(self):
        toolbar = self.addToolBar("主工具栏")
        toolbar.setMovable(False)
        style = self.style()

        self.act_open.setIcon(
            style.standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton))
        self.act_save.setIcon(
            style.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        self.act_submit.setIcon(
            style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.act_cancel.setIcon(
            style.standardIcon(QStyle.StandardPixmap.SP_MediaStop))

        self.act_snapshot = QAction("截图", self)
        self.act_snapshot.setIcon(
            style.standardIcon(QStyle.StandardPixmap.SP_FileDialogContentsView))
        self.act_snapshot.triggered.connect(self._snapshot)

        toolbar.addAction(self.act_open)
        toolbar.addAction(self.act_save)
        toolbar.addSeparator()
        toolbar.addAction(self.act_submit)
        toolbar.addAction(self.act_cancel)
        toolbar.addSeparator()
        toolbar.addAction(self.act_snapshot)

    # ── 状态栏 ──────────────────────────────────────────

    def _init_statusbar(self):
        sb = self.statusBar()
        self.lbl_mol_info = QLabel("分子: (空)  |  原子数: 0")
        self.lbl_calc_status = QLabel("状态: 就绪")
        self.lbl_time = QLabel("耗时: --")
        self.lbl_memory = QLabel("内存: --")
        sb.addWidget(self.lbl_mol_info)
        sb.addWidget(self.lbl_calc_status)
        sb.addWidget(self.lbl_time)
        sb.addPermanentWidget(self.lbl_memory)

    # ── 信号连接 ────────────────────────────────────────

    def _connect_signals(self):
        self.editor.molecule_changed.connect(self._on_molecule_changed)
        self.job_monitor.submit_requested.connect(self._submit_calc)
        self.job_monitor.cancel_requested.connect(self._cancel_calc)
        self.param_panel.params_changed.connect(self._on_params_changed)

    # ════════════════════════════════════════════════════
    #  文件操作
    # ════════════════════════════════════════════════════

    def _open_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "打开分子文件", "",
            "分子文件 (*.xyz *.gjf *.com);;XYZ 文件 (*.xyz);;"
            "Gaussian 输入 (*.gjf *.com);;所有文件 (*)")
        if not filename:
            return
        try:
            from calchem.io.input import read_xyz, read_gjf
            ext = os.path.splitext(filename)[1].lower()
            if ext == ".xyz":
                mol = read_xyz(filename)
            elif ext in (".gjf", ".com"):
                mol = read_gjf(filename)
            else:
                raise ValueError(f"不支持的文件格式: {ext}")

            symbols = [str(s).capitalize() for s in mol.symbols]
            coords = np.array(mol.coordinates, dtype=float).reshape(-1, 3)
            charge = int(getattr(mol, "charge", 0))
            mult = int(getattr(mol, "multiplicity", 1))

            self.editor.set_molecule(symbols, coords)
            self.project.set_molecule(symbols, coords, charge, mult)
            self.project.current_file = filename

            # 同步电荷 / 自旋到参数面板
            try:
                self.param_panel.charge_spin.setValue(charge)
                self.param_panel.mult_spin.setValue(mult)
            except Exception:
                pass

            self._on_molecule_changed(symbols, coords)
            self._update_title()
            self.statusBar().showMessage(f"已打开: {filename}", 3000)
        except Exception as e:
            QMessageBox.critical(self, "打开失败", f"无法读取文件:\n{e}")

    def _save_gjf(self):
        symbols, coords = self.editor.get_molecule()
        if not symbols:
            QMessageBox.warning(self, "警告", "没有分子可保存")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存输入文件", "molecule.gjf",
            "Gaussian 输入 (*.gjf);;所有文件 (*)")
        if not filename:
            return
        try:
            params = self.param_panel.get_params()
            method = params.get("method", "RHF")
            basis = params.get("basis", "sto-3g")
            charge = params.get("charge", 0)
            mult = params.get("multiplicity", 1)
            task = params.get("task_type", "sp")
            task_kw = {"sp": "SP", "opt": "OPT", "freq": "Freq",
                       "excited": "TD"}.get(task, "SP")

            lines = [
                "%chk=calculation.chk",
                f"# {method}/{basis} {task_kw}",
                "",
                "Title Card Required",
                "",
                f"{charge} {mult}",
            ]
            for sym, (x, y, z) in zip(symbols, coords):
                lines.append(f"{sym:<2s} {x:>14.6f} {y:>14.6f} {z:>14.6f}")
            lines.append("")

            with open(filename, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            self.statusBar().showMessage(f"已保存: {filename}", 3000)
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))

    def _export_report(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "导出报告", "calchem_report.txt",
            "文本文件 (*.txt);;所有文件 (*)")
        if not filename:
            return
        try:
            symbols, coords = self.editor.get_molecule()
            params = self.param_panel.get_params()

            lines = ["=" * 60, "CalChem 计算报告", "=" * 60, ""]
            lines.append(f"分子式: {self._formula(symbols)}")
            lines.append(f"原子数: {len(symbols)}")
            lines.append(f"电荷: {params.get('charge', 0)}")
            lines.append(f"自旋多重度: {params.get('multiplicity', 1)}")
            lines.append("")
            lines.append("计算参数:")
            for k, v in params.items():
                lines.append(f"  {k}: {v}")
            lines.append("")
            lines.append("原子坐标 (Å):")
            lines.append(f"{'#':>4} {'元素':>4} {'X':>14} {'Y':>14} {'Z':>14}")
            for i, (sym, coord) in enumerate(zip(symbols, coords)):
                lines.append(
                    f"{i + 1:>4} {sym:>4} "
                    f"{coord[0]:>14.6f} {coord[1]:>14.6f} {coord[2]:>14.6f}"
                )

            results = self.project.results
            if results:
                lines.append("")
                lines.append("计算结果:")
                for k, v in results.items():
                    if isinstance(v, (int, float, str, bool)) or v is None:
                        lines.append(f"  {k}: {v}")
                    elif isinstance(v, (list, tuple)) and len(v) <= 10:
                        lines.append(f"  {k}: {list(v)}")
                    else:
                        lines.append(f"  {k}: <{type(v).__name__}>")

            lines.append("")
            lines.append("=" * 60)

            with open(filename, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            self.statusBar().showMessage(f"报告已导出: {filename}", 3000)
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    # ════════════════════════════════════════════════════
    #  编辑操作
    # ════════════════════════════════════════════════════

    def _add_atom(self):
        try:
            self.editor._on_add_atom()
        except Exception as e:
            QMessageBox.warning(self, "操作失败", str(e))

    def _delete_atom(self):
        try:
            self.editor.remove_selected()
        except Exception as e:
            QMessageBox.warning(self, "操作失败", str(e))

    def _clear_molecule(self):
        try:
            self.editor.set_molecule([], np.zeros((0, 3)))
            self.viewer.clear()
            self.project.symbols = []
            self.project.coordinates = np.zeros((0, 3))
            self._refresh_atom_list([], np.zeros((0, 3)))
            self._update_status()
            self.statusBar().showMessage("分子已清空", 2000)
        except Exception as e:
            QMessageBox.critical(self, "清空失败", str(e))

    def _center_molecule(self):
        try:
            symbols, coords = self.editor.get_molecule()
            if len(symbols) == 0:
                QMessageBox.information(self, "提示", "没有分子可居中")
                return
            com = coords.mean(axis=0)
            centered = coords - com
            self.editor.set_molecule(symbols, centered)
            self._on_molecule_changed(symbols, centered)
            self.statusBar().showMessage("分子已居中至质心", 2000)
        except Exception as e:
            QMessageBox.critical(self, "居中失败", str(e))

    def _insert_fragment(self, frag_name: str):
        """插入分子片段到当前分子末尾。"""
        try:
            if frag_name not in _FRAGMENTS:
                QMessageBox.warning(self, "错误", f"未知片段: {frag_name}")
                return

            frag = _FRAGMENTS[frag_name]
            frag_symbols = frag["symbols"]
            frag_coords = np.array(frag["coords"], dtype=float)

            # 获取当前分子
            curr_symbols, curr_coords = self.editor.get_molecule()

            if len(curr_symbols) == 0:
                # 如果当前没有分子，直接设置片段
                new_symbols = frag_symbols
                new_coords = frag_coords
            else:
                # 添加到当前分子末尾
                # 计算偏移量：将片段放在当前分子右侧 3.0 Å 处
                curr_coords = np.array(curr_coords, dtype=float)
                max_x = curr_coords[:, 0].max() if len(curr_coords) > 0 else 0.0
                frag_max_x = frag_coords[:, 0].max() if len(frag_coords) > 0 else 0.0
                offset_x = max_x + 3.0 - frag_coords[:, 0].min()

                # 平移片段坐标
                frag_coords_shifted = frag_coords.copy()
                frag_coords_shifted[:, 0] += offset_x

                # 合并
                new_symbols = list(curr_symbols) + list(frag_symbols)
                new_coords = np.vstack([curr_coords, frag_coords_shifted])

            # 更新分子编辑器
            self.editor.set_molecule(new_symbols, new_coords)
            self._on_molecule_changed(new_symbols, new_coords)

            self.statusBar().showMessage(f"已插入片段: {frag_name}", 3000)
        except Exception as e:
            QMessageBox.critical(self, "插入失败", str(e))

    # ════════════════════════════════════════════════════
    #  计算操作
    # ════════════════════════════════════════════════════

    def _submit_calc(self):
        try:
            if self.worker is not None and self.worker.isRunning():
                QMessageBox.warning(self, "提示", "已有计算正在运行")
                return

            symbols, coords = self.editor.get_molecule()
            if not symbols:
                QMessageBox.warning(self, "警告", "请先加载或构建分子")
                return

            params = self.param_panel.get_params()
            params["symbols"] = list(symbols)
            params["coordinates"] = np.array(coords, dtype=float)

            # 清空先前结果
            self.conv_plot.clear()
            self.job_monitor.clear_log()
            self.result_viewer.clear()
            self.project.clear_results()
            self._cycle_count = 0

            # 记录开始时间
            self._calc_start_time = time.time()
            self._calc_elapsed_time = None
            self._update_time_display()

            # 创建并启动工作线程
            self.worker = CalcWorker(params, parent=self)
            self.worker.calc_started.connect(self._on_calc_started)
            self.worker.log_received.connect(self._on_log_received)
            self.worker.energy_updated.connect(self._on_energy_updated)
            self.worker.calc_finished.connect(self._on_calc_finished)
            self.worker.calc_error.connect(self._on_calc_error)

            self.project.is_running = True
            self.worker.start()
        except Exception as e:
            self.project.is_running = False
            QMessageBox.critical(self, "提交失败", str(e))

    def _cancel_calc(self):
        try:
            if self.worker is not None and self.worker.isRunning():
                self.worker.cancel()
                self.job_monitor.append_log("\n[用户终止计算]")
                self.job_monitor.set_status(False, False)
                self.project.is_running = False
                self._update_status()
                self._update_time_display()
        except Exception as e:
            QMessageBox.warning(self, "终止失败", str(e))

    def _clear_results(self):
        try:
            self.result_viewer.clear()
            self.conv_plot.clear()
            self.project.clear_results()
            self.job_monitor.clear_log()
            self._calc_elapsed_time = None
            self._update_status()
            self._update_time_display()
            self.statusBar().showMessage("结果已清空", 2000)
        except Exception as e:
            QMessageBox.warning(self, "清空失败", str(e))

    # ════════════════════════════════════════════════════
    #  视图操作
    # ════════════════════════════════════════════════════

    def _on_toggle_style(self, checked: bool):
        # checked=True -> 比例模型 (ball_stick=False)
        try:
            self.viewer.set_style(not checked)
        except Exception:
            pass

    def _on_toggle_axes(self, checked: bool):
        try:
            self.viewer._show_axes = checked
            self.viewer._axes_btn.setText(
                "隐藏坐标轴" if checked else "显示坐标轴")
            self.viewer._update_axes()
        except Exception:
            pass

    def _on_toggle_log_plot(self, checked: bool):
        try:
            self.conv_plot.set_log_scale(checked)
        except Exception:
            pass

    def _on_toggle_monitor(self, checked: bool):
        self.job_monitor.setVisible(checked)

    def _snapshot(self):
        try:
            pixmap = self.viewer.get_snapshot()
            if pixmap.isNull():
                QMessageBox.warning(self, "截图失败", "无法获取截图")
                return
            filename, _ = QFileDialog.getSaveFileName(
                self, "保存截图", "molecule.png",
                "PNG 图片 (*.png);;JPEG 图片 (*.jpg);;所有文件 (*)")
            if filename:
                pixmap.save(filename)
                self.statusBar().showMessage(f"截图已保存: {filename}", 3000)
        except Exception as e:
            QMessageBox.critical(self, "截图失败", str(e))

    # ════════════════════════════════════════════════════
    #  分子变化与参数变化
    # ════════════════════════════════════════════════════

    def _on_molecule_changed(self, symbols, coords):
        try:
            symbols = [str(s) for s in symbols]
            coords = np.asarray(coords, dtype=float).reshape(-1, 3)
            self.viewer.set_molecule(symbols, coords)
            self.project.symbols = symbols
            self.project.coordinates = coords
            self._refresh_atom_list(symbols, coords)
            self._update_status()
        except Exception:
            pass

    def _on_params_changed(self, params):
        self.project.method = params.get("method", "RHF")
        self.project.basis = params.get("basis", "sto-3g")
        self.project.charge = params.get("charge", 0)
        self.project.multiplicity = params.get("multiplicity", 1)
        self.project.task_type = params.get("task_type", "sp")

    # ════════════════════════════════════════════════════
    #  计算回调
    # ════════════════════════════════════════════════════

    def _on_calc_started(self):
        self._cycle_count = 0
        self.job_monitor.set_status(True, False)
        self.job_monitor.append_log("[计算开始]")
        self._update_status()

    def _on_log_received(self, text):
        if text:
            self.job_monitor.append_log(text.rstrip("\n"))
        self.project.log_text += text

    def _on_energy_updated(self, energy, rms):
        self._cycle_count += 1
        self.conv_plot.add_point(float(energy), float(rms))
        self.job_monitor.set_progress(self._cycle_count, float(energy))
        self.project.energy_history.append(float(energy))
        self.project.rms_history.append(float(rms))
        self._update_time_display()

    def _on_calc_finished(self, results):
        try:
            # 计算耗时
            if self._calc_start_time is not None:
                self._calc_elapsed_time = time.time() - self._calc_start_time

            self.project.results = results
            self.project.converged = bool(results.get("converged", False))
            self.project.is_running = False

            # 传递耗时给结果查看器
            self.result_viewer.display_results(results, self._calc_elapsed_time)

            # 用完整历史数据刷新收敛曲线
            eh = results.get("energy_history", [])
            rh = results.get("rms_history", [])
            if eh:
                self.conv_plot.set_data(list(eh), list(rh))

            self.job_monitor.set_status(False, self.project.converged)
            self.job_monitor.append_log("\n[计算完成]")
            e_tot = results.get("e_tot")
            if e_tot is not None:
                self.job_monitor.append_log(
                    f"总能量: {float(e_tot):.10f} Hartree")

            # 显示耗时
            if self._calc_elapsed_time is not None:
                self.job_monitor.append_log(
                    f"计算耗时: {self._calc_elapsed_time:.2f} 秒")

            self.worker = None
            self._update_status()
            self._update_time_display()
        except Exception as e:
            QMessageBox.critical(self, "结果处理失败", str(e))

    def _on_calc_error(self, msg):
        self.project.is_running = False
        self.job_monitor.set_status(False, False)
        self.job_monitor.append_log(f"\n[错误]\n{msg}")

        # 记录耗时（即使出错）
        if self._calc_start_time is not None:
            self._calc_elapsed_time = time.time() - self._calc_start_time

        self.worker = None
        self._update_status()
        self._update_time_display()
        QMessageBox.critical(self, "计算错误", msg)

    # ════════════════════════════════════════════════════
    #  原子列表与状态栏
    # ════════════════════════════════════════════════════

    def _refresh_atom_list(self, symbols, coords):
        self.atom_list_table.setRowCount(0)
        for i, (sym, coord) in enumerate(zip(symbols, coords)):
            self.atom_list_table.insertRow(i)
            sym_upper = str(sym).upper()
            z = _ATOMIC_NUMBERS.get(sym_upper, "")
            items = [
                str(i + 1), str(sym), str(z),
                f"{coord[0]:.6f}", f"{coord[1]:.6f}", f"{coord[2]:.6f}",
            ]
            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.atom_list_table.setItem(i, col, item)

        self.lbl_formula.setText(
            f"分子式: {self._formula(symbols) or '(空)'}")

    def _update_status(self):
        symbols, _ = self.editor.get_molecule()
        formula = self._formula(symbols) if symbols else "(空)"
        natom = len(symbols)
        self.lbl_mol_info.setText(f"分子: {formula}  |  原子数: {natom}")

        if self.project.is_running:
            status = "计算中..."
        elif self.project.converged:
            status = "已收敛"
        elif self.project.results:
            status = "未收敛"
        else:
            status = "就绪"
        self.lbl_calc_status.setText(f"状态: {status}")

    def _update_memory(self):
        mem = self._get_memory_mb()
        if mem is not None:
            self.lbl_memory.setText(f"内存: {mem:.1f} MB")
        else:
            self.lbl_memory.setText("内存: --")

    def _update_time_display(self):
        """更新耗时显示。"""
        if self.project.is_running and self._calc_start_time is not None:
            # 计算进行中，显示实时耗时
            elapsed = time.time() - self._calc_start_time
            self.lbl_time.setText(f"耗时: {elapsed:.1f} 秒")
        elif self._calc_elapsed_time is not None:
            # 计算已完成，显示最终耗时
            self.lbl_time.setText(f"耗时: {self._calc_elapsed_time:.1f} 秒")
        else:
            self.lbl_time.setText("耗时: --")

    def _update_title(self):
        title = "CalChem - 量子化学计算平台"
        if self.project.current_file:
            title = f"{os.path.basename(self.project.current_file)} - {title}"
        self.setWindowTitle(title)

    # ════════════════════════════════════════════════════
    #  关于与关闭
    # ════════════════════════════════════════════════════

    def _about(self):
        QMessageBox.about(
            self, "关于 CalChem",
            "<h3>CalChem 量子化学计算平台</h3>"
            "<p>基于 PyQt6 的量子化学图形界面软件</p>"
            "<p>支持方法: HF / DFT / MP2 / MP4 / CCSD / CCSD(T) / CASSCF</p>"
            "<p>支持任务: 单点能 / 几何优化 / 频率分析 / 激发态</p>"
            "<p>版本: 1.0</p>")

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "确认退出",
                "计算正在进行，确定要退出吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
            self.worker.cancel()
            self.worker.wait(3000)
        self._mem_timer.stop()
        event.accept()

    # ════════════════════════════════════════════════════
    #  辅助方法
    # ════════════════════════════════════════════════════

    @staticmethod
    def _formula(symbols) -> str:
        """根据元素列表生成 Hill 顺序分子式。"""
        if not symbols:
            return ""
        counts = Counter(str(s).capitalize() for s in symbols)
        parts = []
        if "C" in counts:
            parts.append(("C", counts.pop("C")))
        if "H" in counts:
            parts.append(("H", counts.pop("H")))
        for elem in sorted(counts):
            parts.append((elem, counts[elem]))
        return "".join(f"{e}{n if n > 1 else ''}" for e, n in parts)

    @staticmethod
    def _get_memory_mb():
        """返回当前进程内存占用 (MB)，失败返回 None。"""
        # 优先使用 psutil
        try:
            import psutil
            return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        except ImportError:
            pass
        # Windows 回退: 调用 psapi 获取工作集大小
        try:
            import ctypes
            from ctypes import wintypes

            class _ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            psapi = ctypes.WinDLL("psapi.dll")
            kernel32 = ctypes.WinDLL("kernel32.dll")
            if psapi.GetProcessMemoryInfo(
                    kernel32.GetCurrentProcess(),
                    ctypes.byref(counters), counters.cb):
                return counters.WorkingSetSize / 1024 / 1024
        except Exception:
            pass
        return None