"""CalChem 命令行入口。"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from calchem.api.service import TaskService
from calchem.core import CalculationRequest, MoleculeSpec, ResourceLimits
import calchem.backends  # noqa: F401  触发后端注册

console = Console()


@contextlib.contextmanager
def _suppress_stdout():
    """把计算过程的 stdout print 重定向到 stderr（JSON 模式下用）。"""
    saved_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = saved_stdout


@click.group()
@click.version_option(version="0.1.0", prog_name="CalChem")
def cli():
    """CalChem: 统一量子化学计算平台。"""
    pass


@cli.command()
def backends():
    """列出所有可用后端。"""
    svc = TaskService()
    table = Table(title="CalChem Backends")
    table.add_column("Backend", style="cyan")
    table.add_column("Available")
    table.add_column("Version")
    table.add_column("Methods")
    table.add_column("Basis sets")

    for c in svc.list_backends():
        table.add_row(
            c.name,
            "✓" if c.available else "✗",
            c.version or "-",
            str(len(c.methods)),
            str(len(c.basis_sets)),
        )
    console.print(table)


@cli.command()
def doctor():
    """检查环境与依赖。"""
    console.print("[bold]CalChem Doctor[/bold]\n")

    # Python
    console.print(f"  Python: {sys.version.split()[0]}")

    # 后端
    svc = TaskService()
    for c in svc.list_backends():
        if c.available:
            console.print(f"  [green]✓[/green] {c.name}: {c.version or c.executable_path}")
        else:
            console.print(f"  [red]✗[/red] {c.name}: {c.unavailable_reason}")

    # 数据目录
    console.print(f"  Data root: {svc.data_root}")


@cli.command()
@click.option("--molecule", "-i", help="分子文件路径（.xyz）或 'H2'/'water' 内置测试分子")
@click.option("--backend", "-b", default="native", help="后端：native / gaussian / orca")
@click.option("--method", "-M", default="rhf", help="方法：rhf / b3lyp / ccsd(t) 等")
@click.option("--basis", "-B", default="sto-3g", help="基组：sto-3g / 6-31g(d) / def2-svp 等")
@click.option("--charge", default=0, help="电荷")
@click.option("--multiplicity", default=1, help="自旋多重度")
@click.option("--json", "output_json", is_flag=True, help="输出 JSON")
def run(molecule, backend, method, basis, charge, multiplicity, output_json):
    """运行单点能计算。"""
    svc = TaskService()

    # 解析分子
    if molecule is None:
        # 默认 H2 测试
        b2a = 0.529177210903
        mol = MoleculeSpec(
            symbols=["H", "H"],
            coordinates_angstrom=[[0, 0, 0], [0, 0, 1.4 * b2a]],
            charge=charge, multiplicity=multiplicity,
        )
    elif molecule.lower() in ("h2", "water", "h2o"):
        if molecule.lower() == "h2":
            b2a = 0.529177210903
            mol = MoleculeSpec(
                symbols=["H", "H"],
                coordinates_angstrom=[[0, 0, 0], [0, 0, 1.4 * b2a]],
                charge=charge, multiplicity=multiplicity,
            )
        else:
            # H2O Szabo geometry
            b2a = 0.529177210903
            mol = MoleculeSpec(
                symbols=["O", "H", "H"],
                coordinates_angstrom=[
                    [0, 0, 0],
                    [0, 1.432 * b2a, 0],
                    [1.112 * b2a, -0.716 * b2a, 0],
                ],
                charge=charge, multiplicity=multiplicity,
            )
    else:
        # 读 xyz 文件
        p = Path(molecule)
        if not p.exists():
            console.print(f"[red]文件不存在: {molecule}[/red]")
            sys.exit(1)
        lines = p.read_text().strip().split("\n")
        natoms = int(lines[0].strip())
        atoms = []
        coords = []
        for line in lines[2:2 + natoms]:
            parts = line.split()
            atoms.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        mol = MoleculeSpec(symbols=atoms, coordinates_angstrom=coords,
                           charge=charge, multiplicity=multiplicity)

    req = CalculationRequest(
        task="single_point", molecule=mol, backend=backend,
        method=method, basis=basis,
        resources=ResourceLimits(max_cpu_cores=8, max_memory_mb=4000, timeout_seconds=300),
    )

    # --json 模式：状态信息走 stderr，stdout 只输出 JSON
    if output_json:
        click.echo(f"Running: {backend}/{method}/{basis} on {mol.natom} atoms", err=True)
        with _suppress_stdout():
            result = svc.run(req)
        click.echo(result.model_dump_json(indent=2))
    else:
        console.print(f"[cyan]Running:[/cyan] {backend} / {method} / {basis} on {mol.natom} atoms")
        result = svc.run(req)
        console.print(f"\n  Status: [bold]{result.status}[/bold]")
        if result.scf_energy_hartree is not None:
            console.print(f"  Energy: [green]{result.scf_energy_hartree:.10f}[/green] Eh")
        if result.walltime_seconds is not None:
            console.print(f"  Time:   {result.walltime_seconds:.1f}s")
        if result.errors:
            for e in result.errors:
                console.print(f"  [red]Error: {e}[/red]")


if __name__ == "__main__":
    cli()
