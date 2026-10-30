# Windows 11 构建与打包操作手册 (零 C 盘依赖)

## 1. 环境变量重定向 (必须在编译前配置)
在 PowerShell (管理员或用户权限) 中运行：
```powershell
[Environment]::SetEnvironmentVariable("RUSTUP_HOME", "D:\YDW\RustToolchain\rustup", "User")
[Environment]::SetEnvironmentVariable("CARGO_HOME", "D:\YDW\RustToolchain\cargo", "User")
[Environment]::SetEnvironmentVariable("CARGO_TARGET_DIR", "E:\YDW\Trae_Gaussian\build_target", "User")
[Environment]::SetEnvironmentVariable("TMP", "D:\YDW\Temp", "User")
[Environment]::SetEnvironmentVariable("TEMP", "D:\YDW\Temp", "User")
[Environment]::SetEnvironmentVariable("RUSTFLAGS", "-C target-cpu=znver5 -C target-feature=+avx512f,+avx512dq,+avx512vl,+fma", "User")
```

## 2. 编译并安装 Rust 核心扩展模块
使用 `molforge-webengine` 环境：
```powershell
$py = "D:\YDW\Trae_Gromacs\miniconda3\envs\molforge-webengine\python.exe"
cd E:\YDW\Trae_Gaussian

# 1. 运行 Rust 单元测试 (含数值基线验证)
cargo test --workspace --release

# 2. 编译并安装为 Python 扩展模块
& $py -m maturin develop --release
```

## 3. 运行 Python 单元测试回归
```powershell
& $py -m pytest tests/unit -v
```

## 4. PyInstaller One-Folder 独立目录打包
```powershell
& $py -m PyInstaller --noconfirm --onedir `
    --workpath "E:\YDW\Trae_Gaussian\build_pyinstaller" `
    --distpath "E:\YDW\Trae_Gaussian\dist_onefolder" `
    --add-data "crates/calchem-basis/data;calchem/basis_data" `
    --hidden-import "calchem._core" `
    E:\YDW\Trae_Chemdraw\MolForge\run_molforge.py
```
