from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from ..utils.constants import __version__
from ..utils.logging_utils import setup_logging, get_logger
from ..utils.config import get_config, Config

console = Console()
logger = get_logger(__name__)


@click.group()
@click.version_option(version=__version__, prog_name="calchem")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress output")
@click.option("--config", type=click.Path(), help="Path to config file")
@click.pass_context
def cli(ctx, verbose, quiet, config):
    """CalChem - GPU-accelerated quantum chemistry software"""
    ctx.ensure_object(dict)

    if quiet:
        log_level = "ERROR"
    elif verbose:
        log_level = "DEBUG"
    else:
        log_level = "INFO"

    setup_logging(level=log_level)

    if config:
        cfg = Config.load(config)
        from ..utils.config import set_config

        set_config(cfg)


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--method", "-m", default="hf", help="Method: hf, dft, mp2, etc.")
@click.option("--basis", "-b", default="sto-3g", help="Basis set")
@click.option("--xc", default="b3lyp", help="DFT functional")
@click.option("--charge", "-c", default=0, type=int, help="Molecular charge")
@click.option("--multiplicity", "-s", default=1, type=int, help="Spin multiplicity")
@click.option("--output", "-o", type=click.Path(), help="Output file (JSON)")
def run(input_file, method, basis, xc, charge, multiplicity, output):
    """Run a quantum chemistry calculation"""
    from ..api.high_level import single_point

    console.print(f"[bold blue]CalChem {__version__}[/bold blue]")
    console.print(f"Input: {input_file}")
    console.print(f"Method: {method.upper()}, Basis: {basis}")

    try:
        result = single_point(
            input_file,
            method=method,
            basis=basis,
            xc=xc,
            charge=charge,
            multiplicity=multiplicity,
        )

        console.print("\n[bold green]Calculation complete![/bold green]")
        console.print(f"  Energy: [yellow]{result.energy:.10f}[/yellow] Eh")
        console.print(f"  Converged: {result.converged}")
        console.print(f"  Cycles: {result.n_cycles}")

        if output:
            result.save(output)
            console.print(f"\nResults saved to: {output}")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


@cli.command("opt")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--method", "-m", default="hf", help="Method: hf, dft, mp2, etc.")
@click.option("--basis", "-b", default="sto-3g", help="Basis set")
@click.option("--xc", default="b3lyp", help="DFT functional")
@click.option("--charge", "-c", default=0, type=int, help="Molecular charge")
@click.option("--multiplicity", "-s", default=1, type=int, help="Spin multiplicity")
@click.option("--optimizer", default="bfgs", help="Optimizer: bfgs, berny")
@click.option("--output", "-o", type=click.Path(), help="Output file (XYZ)")
def optimize(input_file, method, basis, xc, charge, multiplicity, optimizer, output):
    """Run a geometry optimization"""
    from ..api.high_level import optimize

    console.print(f"[bold blue]CalChem {__version__} - Geometry Optimization[/bold blue]")
    console.print(f"Input: {input_file}")
    console.print(f"Method: {method.upper()}, Basis: {basis}, Optimizer: {optimizer}")

    try:
        result_mol, energy = optimize(
            input_file,
            method=method,
            basis=basis,
            xc=xc,
            charge=charge,
            multiplicity=multiplicity,
            optimizer=optimizer,
        )

        console.print("\n[bold green]Optimization complete![/bold green]")
        console.print(f"  Final energy: [yellow]{energy:.10f}[/yellow] Eh")

        if output:
            result_mol.to_xyz_file(output, comment=f"Optimized at {method.upper()}/{basis}")
            console.print(f"  Optimized geometry saved to: {output}")
        else:
            console.print("\nOptimized geometry:")
            console.print(result_mol.to_xyz())

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


@cli.command("freq")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--method", "-m", default="hf", help="Method: hf, dft")
@click.option("--basis", "-b", default="sto-3g", help="Basis set")
@click.option("--xc", default="b3lyp", help="DFT functional")
@click.option("--charge", "-c", default=0, type=int, help="Molecular charge")
@click.option("--multiplicity", "-s", default=1, type=int, help="Spin multiplicity")
@click.option("--temperature", default=298.15, type=float, help="Temperature (K)")
def frequency(input_file, method, basis, xc, charge, multiplicity, temperature):
    """Run a frequency analysis"""
    from ..api.high_level import frequency

    console.print(f"[bold blue]CalChem {__version__} - Frequency Analysis[/bold blue]")
    console.print(f"Input: {input_file}")
    console.print(f"Method: {method.upper()}, Basis: {basis}")
    console.print(f"Temperature: {temperature} K")

    try:
        freqs, thermo = frequency(
            input_file,
            method=method,
            basis=basis,
            xc=xc,
            charge=charge,
            multiplicity=multiplicity,
            temperature=temperature,
        )

        console.print("\n[bold green]Frequency analysis complete![/bold green]")
        console.print(f"\nVibrational frequencies (cm^-1):")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Mode", justify="right")
        table.add_column("Frequency (cm⁻¹)", justify="right")

        for i, f in enumerate(freqs):
            table.add_row(str(i + 1), f"{f:.2f}")

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


@cli.command("cis")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--basis", "-b", default="sto-3g", help="Basis set")
@click.option("--nstates", "-n", default=10, type=int, help="Number of excited states")
@click.option("--charge", "-c", default=0, type=int, help="Molecular charge")
@click.option("--multiplicity", "-s", default=1, type=int, help="Spin multiplicity")
@click.option("--triplet", is_flag=True, help="Compute triplet states instead of singlet")
def cis(input_file, basis, nstates, charge, multiplicity, triplet):
    """Run a CIS excited state calculation"""
    from ..api.high_level import excited

    console.print(f"[bold blue]CalChem {__version__} - CIS Excited States[/bold blue]")
    console.print(f"Input: {input_file}")
    console.print(f"Basis: {basis}, States: {nstates}")
    console.print(f"Spin: {'triplet' if triplet else 'singlet'}")

    try:
        result = excited(
            input_file,
            method="cis",
            basis=basis,
            nstates=nstates,
            singlet=not triplet,
            charge=charge,
            multiplicity=multiplicity,
        )

        console.print("\n[bold green]CIS calculation complete![/bold green]")
        console.print(f"  Ground state energy: {result['ground_state_energy']:.6f} Eh")

        console.print(f"\nExcitation energies:")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("State", justify="right")
        table.add_column("Energy (eV)", justify="right")
        table.add_column("Osc. Strength", justify="right")

        e_exc = result["excitation_energies"]
        osc = result["oscillator_strengths"]

        for i in range(len(e_exc)):
            e_ev = e_exc[i] * 27.2114
            table.add_row(f"S{i+1}", f"{e_ev:.4f}", f"{osc[i]:.6f}")

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
def info(input_file):
    """Display information about a molecule"""
    from ..io.input import read_molecule

    try:
        mol = read_molecule(input_file)

        console.print(f"[bold blue]Molecule Information[/bold blue]")
        console.print(f"  Formula: [cyan]{mol.formula}[/cyan]")
        console.print(f"  Atoms: {mol.natom}")
        console.print(f"  Charge: {mol.charge}")
        console.print(f"  Multiplicity: {mol.multiplicity}")
        console.print(f"  Units: {mol.units}")
        console.print(f"  Electrons: {mol.nelectron}")

        if mol.natom <= 20:
            console.print(f"\n{mol.to_xyz()}")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


@cli.command()
@click.argument("filename", type=click.Path())
@click.option("--name", "-n", help="Config name to set")
@click.option("--value", "-v", help="Config value to set")
def config(filename, name, value):
    """Manage configuration"""
    if Path(filename).exists():
        cfg = Config.load(filename)
        console.print(f"[bold]Current configuration:[/bold]")
        for k, v in cfg.model_dump().items():
            console.print(f"  {k}: {v}")
    else:
        if click.confirm(f"Create new config file at {filename}?"):
            cfg = Config()
            cfg.save(filename)
            console.print(f"Default config saved to {filename}")


@cli.command()
def basis():
    """List available basis sets"""
    from ..basis.library import list_available_basis

    basis_sets = list_available_basis()

    console.print("[bold blue]Available basis sets:[/bold blue]")
    for b in basis_sets:
        console.print(f"  - {b}")


if __name__ == "__main__":
    cli()
