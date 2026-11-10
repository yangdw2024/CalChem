# 运行目录位置

Python 对本 docs 审计路径直接写入曾被拒绝。本轮实际运行目录为：

E:\YDW\Trae_Gaussian\build_target\runtime\redteam_20261109

其中存放独立重建扩展、Rust 探针程序、Numba/pytest 缓存、外部程序工作目录与隔离的 wheel 构建暂存树。没有将该问题通过修改权限或业务代码解决。可保留的主要输入/输出已复制到审计 evidence/，该文件仅用于目录定位。
