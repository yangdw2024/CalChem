"""Test methods through GUI - with flush"""
import sys, os, time
sys.path.insert(0, r'D:\YDW\CalChem\gui')
sys.path.insert(0, r'D:\YDW\Trae_Gaussian')

import numpy as np
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import QObject, pyqtSignal

app = QApplication(sys.argv)
QMessageBox.critical = staticmethod(lambda *a, **kw: None)
QMessageBox.warning = staticmethod(lambda *a, **kw: None)
QMessageBox.information = staticmethod(lambda *a, **kw: None)

from windows.main_window import MainWindow

w = MainWindow()
symbols = ['O', 'H', 'H']
coords = np.array([[0.0,0.0,0.0],[0.757,0.586,0.0],[-0.757,0.586,0.0]])
w.editor.set_molecule(symbols, coords)

class Tracker(QObject):
    fin = pyqtSignal(dict)
    err = pyqtSignal(str)
tracker = Tracker()
received = {}
tracker.fin.connect(lambda r: received.update({'fin': r}))
tracker.err.connect(lambda m: received.update({'err': m}))

def run_test(method):
    received.clear()
    combo = w.param_panel.method_combo
    idx = combo.findText(method)
    if idx >= 0:
        combo.setCurrentIndex(idx)

    w._submit_calc()
    if w.worker:
        w.worker.calc_finished.connect(tracker.fin)
        w.worker.calc_error.connect(tracker.err)

    elapsed = 0
    while w.worker is not None and w.worker.isRunning() and elapsed < 300:
        app.processEvents()
        time.sleep(0.05)
        elapsed += 0.05

    for _ in range(10):
        app.processEvents()
        time.sleep(0.02)

    if 'fin' in received:
        r = received['fin']
        e = r.get('e_tot', 0)
        c = r.get('converged', False)
        n = r.get('n_cycles', 0)
        msg = f"  {method:12s}: E={e:.8f}  conv={c}  cycles={n}  {elapsed:.1f}s"
    elif 'err' in received:
        msg = f"  {method:12s}: ERROR - {received['err'][:200]}"
    else:
        msg = f"  {method:12s}: NO SIGNAL ({elapsed:.1f}s)"
    print(msg, flush=True)

print("=" * 60, flush=True)
print("CalChem GUI Multi-Method Test (H2O/sto-3g)", flush=True)
print("=" * 60, flush=True)

for method in ["RHF", "PBE"]:
    run_test(method)
    app.processEvents()

print("=" * 60, flush=True)
print("Fast methods done!", flush=True)
