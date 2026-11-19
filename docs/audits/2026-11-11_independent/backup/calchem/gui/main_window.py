"""CalChem 主窗口。"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction, QColor, QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTabWidget, QTableWidget, QTableWidgetItem, QFormLayout,
    QComboBox, QLineEdit, QPushButton, QLabel, QTextEdit, QGroupBox,
    QHeaderView, QFileDialog, QMessageBox, QStatusBar, QProgressBar,
    QListWidget, QListWidgetItem, QDoubleSpinBox, QSpinBox,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from calchem.api.service import TaskService
from calchem.core import CalculationRequest, MoleculeSpec, ResourceLimits
import calchem.backends  # noqa: F401

# 元素颜色（CPK）
ELEMENT_COLORS = {
    "H": "#FFFFFF", "C": "#909090", "N": "#3050F8", "O": "#FF0D0D",
    "F": "#90E050", "Cl": "#1FF01F", "Br": "#A62929", "I": "#C020C0",
    "S": "#FFFF30", "P": "#FF8000", "B": "#FFB5B5", "Si": "#F0C0A0",
    "Mg": "#8AFF00", "Ca": "#3DFF00", "Fe": "#E06633", "Zn": "#7D80B0",
}
DEFAULT_COLOR = "#D0D0D0"


class Molecule3DWidget(FigureCanvas):
    """matplotlib 3D 分子视图。"""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 4), dpi=100)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self._mol = None
        self._draw_molecule()

    def set_molecule(self, mol: MoleculeSpec):
        self._mol = mol
        self._draw_molecule()

    def _draw_molecule(self):
        self.ax.clear()
        if self._mol is None:
            self.ax.text(0, 0, 0, "加载分子以显示", ha="center")
            self.draw()
            return

        coords = np.array(self._mol.coordinates_angstrom)
        symbols = self._mol.symbols

        for i, (sym, xyz) in enumerate(zip(symbols, coords)):
            color = ELEMENT_COLORS.get(sym, DEFAULT_COLOR)
            size = 200 if sym == "H" else 400
            self.ax.scatter(xyz[0], xyz[1], xyz[2], c=color, s=size,
                           edgecolors="black", linewidths=0.5, depthshade=True)
            self.ax.text(xyz[0], xyz[1], xyz[2], sym, fontsize=8, ha="center")

        # 画键（距离 < 1.9 Å 视为键）
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                d = np.linalg.norm(coords[i] - coords[j])
                if d < 1.9:
                    self.ax.plot([coords[i, 0], coords[j, 0]],
                                [coords[i, 1], coords[j, 1]],
                                [coords[i, 2], coords[j, 2]],
                                color="gray", linewidth=2, alpha=0.6)

        self.ax.set_xlabel("X (Å)")
        self.ax.set_ylabel("Y (Å)")
        self.ax.set_zlabel("Z (Å)")
        self.ax.set_box_aspect([1, 1, 1])
        self.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CalChem - 统一量子化学平台")
        self.resize(1440, 900)

        self.service = TaskService()
        self.current_mol: Optional[MoleculeSpec] = None
        self.current_result: Optional[dict] = None
        self._threads = []  # 防止 GC

        self._build_ui()
        self._refresh_backends()
        self._load_default_molecule()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # 顶栏
        topbar = QHBoxLayout()
        topbar.addWidget(QLabel("<b>CalChem</b>"))
        topbar.addStretch()
        self.backend_status_label = QLabel("后端检测中...")
        topbar.addWidget(self.backend_status_label)
        layout.addLayout(topbar)

        # 主体三栏
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左：原子表
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("<b>原子表</b>"))
        self.atom_table = QTableWidget(0, 5)
        self.atom_table.setHorizontalHeaderLabels(["#", "元素", "X (Å)", "Y (Å)", "Z (Å)"])
        self.atom_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        left_layout.addWidget(self.atom_table)

        btn_row = QHBoxLayout()
        btn_row.addWidget(QPushButton("导入 XYZ", clicked=self._import_xyz))
        btn_row.addWidget(QPushButton("H2 测试", clicked=lambda: self._load_preset("H2")))
        btn_row.addWidget(QPushButton("H2O 测试", clicked=lambda: self._load_preset("H2O")))
        left_layout.addLayout(btn_row)
        splitter.addWidget(left)

        # 中：3D 视图
        center = QWidget()
        center_layout = QVBoxLayout(center)
        self.mol_view = Molecule3DWidget()
        center_layout.addWidget(NavigationToolbar(self.mol_view, self))
        center_layout.addWidget(self.mol_view)
        splitter.addWidget(center)

        # 右：设置 + 结果
        right_tabs = QTabWidget()

        # 计算设置 tab
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)

        form = QFormLayout()
        self.backend_combo = QComboBox()
        form.addRow("后端", self.backend_combo)
        self.method_combo = QComboBox()
        self.method_combo.addItems(["rhf", "uhf", "b3lyp", "pbe", "mp2", "ccsd", "ccsd(t)"])
        form.addRow("方法", self.method_combo)
        self.basis_combo = QComboBox()
        self.basis_combo.addItems(["sto-3g", "3-21g", "6-31g", "6-31g(d)", "6-311g(d,p)", "def2-svp", "def2-tzvp"])
        form.addRow("基组", self.basis_combo)
        self.charge_spin = QSpinBox()
        self.charge_spin.setValue(0)
        form.addRow("电荷", self.charge_spin)
        self.mult_spin = QSpinBox()
        self.mult_spin.setValue(1)
        form.addRow("多重度", self.mult_spin)
        self.memory_spin = QSpinBox()
        self.memory_spin.setRange(500, 12000)
        self.memory_spin.setValue(4000)
        self.memory_spin.setSuffix(" MB")
        form.addRow("内存", self.memory_spin)
        settings_layout.addLayout(form)

        self.run_btn = QPushButton("运行计算")
        self.run_btn.setStyleSheet("background-color: #244B78; color: white; padding: 8px; font-weight: bold;")
        self.run_btn.clicked.connect(self._run_calculation)
        settings_layout.addWidget(self.run_btn)

        settings_layout.addWidget(QLabel("<b>计算日志</b>"))
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(150)
        settings_layout.addWidget(self.log_view)

        right_tabs.addTab(settings_widget, "计算设置")

        # 结果 tab
        result_widget = QWidget()
        result_layout = QVBoxLayout(result_widget)
        self.result_view = QTextEdit()
        self.result_view.setReadOnly(True)
        result_layout.addWidget(self.result_view)
        right_tabs.addTab(result_widget, "结果")

        # 后端状态 tab
        backend_widget = QWidget()
        backend_layout = QVBoxLayout(backend_widget)
        self.backend_table = QTableWidget(0, 4)
        self.backend_table.setHorizontalHeaderLabels(["后端", "可用", "版本", "路径"])
        self.backend_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        backend_layout.addWidget(self.backend_table)
        right_tabs.addTab(backend_widget, "后端状态")

        splitter.addWidget(right_tabs)
        splitter.setSizes([280, 560, 360])
        layout.addWidget(splitter, stretch=1)

        # 底部队列
        self.queue_label = QLabel("队列为空")
        layout.addWidget(self.queue_label)

        # 状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("就绪")

    def _refresh_backends(self):
        caps_list = self.service.list_backends()
        self.backend_combo.clear()
        self.backend_table.setRowCount(0)
        for c in caps_list:
            self.backend_combo.addItem(c.name)
            row = self.backend_table.rowCount()
            self.backend_table.insertRow(row)
            self.backend_table.setItem(row, 0, QTableWidgetItem(c.name))
            self.backend_table.setItem(row, 1, QTableWidgetItem("✓" if c.available else "✗"))
            self.backend_table.setItem(row, 2, QTableWidgetItem(c.version or "-"))
            self.backend_table.setItem(row, 3, QTableWidgetItem(c.executable_path or "-"))
        available = sum(1 for c in caps_list if c.available)
        self.backend_status_label.setText(f"可用后端: {available}/{len(caps_list)}")

    def _load_preset(self, name: str):
        b2a = 0.529177210903
        if name == "H2":
            self.current_mol = MoleculeSpec(
                symbols=["H", "H"],
                coordinates_angstrom=[[0, 0, 0], [0, 0, 1.4 * b2a]],
            )
        elif name == "H2O":
            self.current_mol = MoleculeSpec(
                symbols=["O", "H", "H"],
                coordinates_angstrom=[
                    [0, 0, 0],
                    [0, 1.432 * b2a, 0],
                    [1.112 * b2a, -0.716 * b2a, 0],
                ],
            )
        self._update_mol_view()

    def _load_default_molecule(self):
        self._load_preset("H2")

    def _import_xyz(self):
        path, _ = QFileDialog.getOpenFileName(self, "导入 XYZ", "", "XYZ files (*.xyz);;All files (*)")
        if not path:
            return
        try:
            lines = Path(path).read_text().strip().split("\n")
            natoms = int(lines[0].strip())
            symbols, coords = [], []
            for line in lines[2:2 + natoms]:
                parts = line.split()
                symbols.append(parts[0])
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            self.current_mol = MoleculeSpec(symbols=symbols, coordinates_angstrom=coords)
            self._update_mol_view()
            self.statusBar.showMessage(f"已导入: {path}")
        except Exception as e:
            QMessageBox.critical(self, "导入失败", str(e))

    def _update_mol_view(self):
        if self.current_mol is None:
            return
        self.mol_view.set_molecule(self.current_mol)
        self.atom_table.setRowCount(self.current_mol.natom)
        for i, (sym, xyz) in enumerate(zip(self.current_mol.symbols, self.current_mol.coordinates_angstrom)):
            self.atom_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.atom_table.setItem(i, 1, QTableWidgetItem(sym))
            self.atom_table.setItem(i, 2, QTableWidgetItem(f"{xyz[0]:.6f}"))
            self.atom_table.setItem(i, 3, QTableWidgetItem(f"{xyz[1]:.6f}"))
            self.atom_table.setItem(i, 4, QTableWidgetItem(f"{xyz[2]:.6f}"))

    def _run_calculation(self):
        if self.current_mol is None:
            QMessageBox.warning(self, "错误", "请先加载分子")
            return

        backend = self.backend_combo.currentText()
        method = self.method_combo.currentText()
        basis = self.basis_combo.currentText()

        self.current_mol.charge = self.charge_spin.value()
        self.current_mol.multiplicity = self.mult_spin.value()

        req = CalculationRequest(
            task="single_point", molecule=self.current_mol,
            backend=backend, method=method, basis=basis,
            resources=ResourceLimits(
                max_cpu_cores=8,
                max_memory_mb=self.memory_spin.value(),
                timeout_seconds=300,
            ),
        )

        self.run_btn.setEnabled(False)
        self.statusBar.showMessage("计算中...")
        self.log_view.append(f"[{datetime.now().strftime('%H:%M:%S')}] 提交: {backend}/{method}/{basis}")

        from calchem.gui.worker import start_calc_thread
        thread, worker = start_calc_thread(
            self.service, req,
            on_finished=self._on_calc_finished,
            on_failed=self._on_calc_failed,
            on_log=self.log_view.append,
        )
        self._threads.append(thread)

    def _on_calc_finished(self, result_json: str):
        result = json.loads(result_json)
        self.current_result = result
        self.run_btn.setEnabled(True)
        self.statusBar.showMessage("计算完成")

        energy = result.get("scf_energy_hartree")
        status = result.get("status")
        walltime = result.get("walltime_seconds", 0)

        html = f"""<h3>计算结果</h3>
        <p><b>状态:</b> {status}</p>
        <p><b>后端:</b> {result.get('backend')}</p>
        <p><b>方法/基组:</b> {result.get('method')}/{result.get('basis')}</p>
        <p><b>总能量:</b> <span style='font-size:18px;color:#2E8B68'>{energy:.10f} Eh</span></p>
        <p><b>耗时:</b> {walltime:.1f}s</p>
        """
        if result.get("errors"):
            html += f"<p style='color:#B94A48'><b>错误:</b> {'; '.join(result['errors'])}</p>"
        self.result_view.setHtml(html)
        self.log_view.append(f"完成: E={energy:.10f} Eh, status={status}")

    def _on_calc_failed(self, err_type: str, msg: str):
        self.run_btn.setEnabled(True)
        self.statusBar.showMessage("计算失败")
        self.log_view.append(f"错误 [{err_type}]: {msg[:200]}")
        QMessageBox.critical(self, "计算失败", f"{err_type}: {msg[:500]}")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
