use calchem_basis::BasisSet;
use calchem_dft::{D3Model, GridLevel, XcFunctional};
use calchem_model::{symbol_from_atomic_number, Atom, Molecule, ScfOptions, ScfSummary};
use calchem_opt::{GasPoint, OptOptions, Reference};
use calchem_tddft::SpinManifold;
mod checkpoint;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Task {
    Energy,
    Gradient,
    Opt,
    Freq,
    TdDft,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Method {
    Rks,
    Uks,
    Rhf,
    Uhf,
}

#[derive(Debug, Clone)]
struct CliArgs {
    input_path: String,
    task: Task,
    method: Method,
    xc: String,
    basis: String,
    grid: String,
    rij: bool,
    jfit_basis: calchem_model::JFitBasis,
    direct: Option<bool>,
    gpu: bool,
    charge_override: Option<i32>,
    mult_override: Option<usize>,
    max_iter: usize,
    conv_energy: f64,
    conv_density: f64,
    diis_max_vecs: usize,
    solv_model: Option<String>,
    solvent: Option<String>,
    dispersion: Option<String>,
    tda: bool,
    spin: SpinManifold,
    threads: Option<usize>,
    freq_step: f64,
    freq_jobs: usize,
    checkpoint: Option<String>,
    resume: Option<String>,
    optimizer: OptOptions,
    nstates: usize,
    opt_max_cycles: usize,
    json: bool,
    out_path: Option<String>,
}

fn print_usage() {
    eprintln!(
        "CalChem Native Quantum Chemistry Engine\n\
         Usage: calchem [options] <input.gjf|input.xyz>\n\n\
         Task Selection:\n\
         \t--task <energy|grad|opt|freq|tddft>   (default: energy)\n\n\
         Electronic Structure Options:\n\
         \t--method <rks|uks|rhf|uhf>           (default: rks)\n\
         \t--xc <functional>                    (default: b3lyp)\n\
         \t--basis <basis>                      (default: sto-3g)\n\
         \t--grid <coarse|medium|fine|ultrafine> (default: coarse)\n\
         \t--rij                                Enable RI-J density fitting\n\
         \t--gpu                                Enable native CUDA GPU acceleration\n\
         \t--direct                             Enable Direct SCF (on-the-fly ERIs)\n\
         \t--stored                             Disable Direct SCF (use stored ERIs)\n\
         \t--charge <int>                       Override net molecular charge\n\
         \t--mult <uint>                        Override spin multiplicity\n\
         \t--max-iter <uint>                    SCF iteration limit (default: 100)\n\
         \t--conv-e <float>                     Energy convergence threshold (default: 1e-10)\n\
         \t--conv-d <float>                     Density convergence threshold (default: 1e-8)\n\
         \t--diis-max-vecs <uint>               DIIS subspace size (default: 6)\n\n\
         Physical Environments & Properties:\n\
         \t--solv <pcm|iefpcm|cpcm|smd>         Implicit continuum solvation model (SCRF)\n\
         \t--solvent <name>                     Solvent name (e.g. water, chloroform)\n\
         \t--disp <d3bj>                        Grimme D3(BJ) dispersion correction\n\
         \t--nstates <uint>                     Excited states count for TD-DFT (default: 3)\n\
         \t--opt-max-cycles <uint>              Max geometry optimization steps (default: 50)\n\n\
         --aux-basis <auto-jfit|def2-jfit-f> (Cartesian auxiliary shells through f)\n\
         --threads N  --td-mode <tda|lr>  --spin <singlet|triplet|both>\n\
         --freq-step BOHR  --freq-jobs N (concurrent displacement SCFs, default 1)\n\
         --max-force F  --rms-force F  --max-displacement BOHR  --max-step-bohr BOHR\n\
         --max-line-searches N  --checkpoint FILE  --resume FILE\n\
         Output Formats:\n\
         \t--json                               Emit structured machine-readable JSON\n\
         \t--out <file>                         Write output to specified file\n\
         \t-h, --help                           Show this help message"
    );
}

fn positive_count(s: &str, flag: &str) -> Result<usize,String> {
    s.parse::<usize>().ok().filter(|v| *v > 0).ok_or_else(||format!("{flag} needs a positive integer"))
}
fn positive_float(s: &str, flag: &str) -> Result<f64,String> {
    s.parse::<f64>().ok().filter(|v| v.is_finite() && *v > 0.0).ok_or_else(||format!("{flag} needs a positive number"))
}

fn parse_args() -> Result<CliArgs, String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        print_usage();
        std::process::exit(if args.is_empty() { 2 } else { 0 });
    }

    let mut input_path: Option<String> = None;
    let mut task = Task::Energy;
    let mut method = Method::Rks;
    let mut xc = "b3lyp".to_string();
    let mut basis = "sto-3g".to_string();
    let mut grid = "coarse".to_string();
    let mut jfit_basis = calchem_model::JFitBasis::Auto;
    let mut rij = false;
    let mut direct: Option<bool> = None;
    let mut gpu = false;
    let mut charge_override = None;
    let mut mult_override = None;
    let mut max_iter = 100usize;
    let mut conv_energy = 1e-10_f64;
    let mut conv_density = 1e-8_f64;
    let mut diis_max_vecs = 6usize;
    let mut solv_model = None;
    let mut solvent = None;
    let mut dispersion = None;
    let mut tda = true;
    let mut spin = SpinManifold::Singlet;
    let mut threads = None;
    let mut freq_step = 0.001;
    let mut freq_jobs = 1;
    let mut checkpoint = None;
    let mut resume = None;
    let mut optimizer = OptOptions::default();
    let mut nstates = 3usize;
    let mut opt_max_cycles = 50usize;
    let mut json = false;
    let mut out_path = None;

    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        let next_val = |idx: &mut usize, flag: &str| -> Result<String, String> {
            *idx += 1;
            args.get(*idx)
                .cloned()
                .ok_or_else(|| format!("Missing value for {flag}"))
        };

        match arg.as_str() {
            "--task" => {
                let v = next_val(&mut i, "--task")?.to_ascii_lowercase();
                task = match v.as_str() {
                    "energy" | "sp" => Task::Energy,
                    "grad" | "gradient" | "force" => Task::Gradient,
                    "opt" | "optimize" | "optgeom" => Task::Opt,
                    "freq" | "frequency" | "thermo" => Task::Freq,
                    "tddft" | "uvvis" | "excited" => Task::TdDft,
                    _ => return Err(format!("Unsupported task '{v}'")),
                };
            }
            "--method" => {
                let v = next_val(&mut i, "--method")?.to_ascii_lowercase();
                method = match v.as_str() {
                    "rks" => Method::Rks,
                    "uks" => Method::Uks,
                    "rhf" => Method::Rhf,
                    "uhf" => Method::Uhf,
                    _ => return Err(format!("Unsupported method '{v}'")),
                };
            }
            "--xc" => xc = next_val(&mut i, "--xc")?,
            "--basis" => basis = next_val(&mut i, "--basis")?,
            "--grid" => grid = next_val(&mut i, "--grid")?,
            "--rij" => rij = true,
            "--aux-basis" => {
                rij = true;
                jfit_basis = match next_val(&mut i,arg)?.as_str() {
                    "auto-jfit" => calchem_model::JFitBasis::Auto,
                    "def2-jfit-f" => calchem_model::JFitBasis::Def2UpToF,
                    v => return Err(format!("Unknown RI-J auxiliary basis '{v}'")),
                }
            }
            "--gpu" => gpu = true,
            "--direct" => direct = Some(true),
            "--stored" => direct = Some(false),
            "--charge" => {
                let v = next_val(&mut i, "--charge")?;
                charge_override = Some(v.parse::<i32>().map_err(|e| format!("Invalid --charge '{v}': {e}"))?);
            }
            "--mult" => {
                let v = next_val(&mut i, "--mult")?;
                mult_override = Some(v.parse::<usize>().map_err(|e| format!("Invalid --mult '{v}': {e}"))?);
            }
            "--max-iter" => {
                let v = next_val(&mut i, "--max-iter")?;
                max_iter = positive_count(&v, "--max-iter")?;
            }
            "--conv-e" => {
                let v = next_val(&mut i, "--conv-e")?;
                conv_energy = positive_float(&v, "--conv-e")?;
            }
            "--conv-d" => {
                let v = next_val(&mut i, "--conv-d")?;
                conv_density = positive_float(&v, "--conv-d")?;
            }
            "--diis-max-vecs" => {
                let v = next_val(&mut i, "--diis-max-vecs")?;
                diis_max_vecs = positive_count(&v, "--diis-max-vecs")?;
            }
            "--solv" => solv_model = Some(next_val(&mut i, "--solv")?.to_ascii_lowercase()),
            "--solvent" => solvent = Some(next_val(&mut i, "--solvent")?.to_ascii_lowercase()),
            "--disp" => dispersion = Some(next_val(&mut i, "--disp")?.to_ascii_lowercase()),
            "--td-mode" => tda = match next_val(&mut i,arg)?.as_str() { "tda" => true, "lr" | "rpa" => false, v => return Err(format!("Invalid TD mode '{v}'")) },
            "--spin" => spin = SpinManifold::from_name(&next_val(&mut i,arg)?).map_err(|e| e.to_string())?,
            "--threads" => threads = Some(positive_count(&next_val(&mut i,arg)?,arg)?),
            "--freq-jobs" => freq_jobs = positive_count(&next_val(&mut i,arg)?,arg)?,
            "--freq-step" => freq_step = positive_float(&next_val(&mut i,arg)?,arg)?,
            "--max-force" => optimizer.max_force = positive_float(&next_val(&mut i,arg)?,arg)?,
            "--rms-force" => optimizer.rms_force = positive_float(&next_val(&mut i,arg)?,arg)?,
            "--max-displacement" => optimizer.max_displacement = positive_float(&next_val(&mut i,arg)?,arg)?,
            "--max-step-bohr" => optimizer.max_step_bohr = positive_float(&next_val(&mut i,arg)?,arg)?,
            "--max-line-searches" => optimizer.max_line_searches = positive_count(&next_val(&mut i,arg)?,arg)?,
            "--checkpoint" => checkpoint = Some(next_val(&mut i,arg)?),
            "--resume" => resume = Some(next_val(&mut i,arg)?),
            "--nstates" => {
                let v = next_val(&mut i, "--nstates")?;
                nstates = positive_count(&v, "--nstates")?;
            }
            "--opt-max-cycles" => {
                let v = next_val(&mut i, "--opt-max-cycles")?;
                opt_max_cycles = positive_count(&v, "--opt-max-cycles")?;
            }
            "--json" => json = true,
            "--out" => out_path = Some(next_val(&mut i, "--out")?),
            s if s.starts_with('-') => return Err(format!("Unknown option flag '{s}'")),
            pos => {
                if input_path.is_some() {
                    return Err(format!("Unexpected extra positional argument '{pos}'"));
                }
                input_path = Some(pos.to_string());
            }
        }
        i += 1;
    }

    let input_path = input_path.ok_or_else(|| "Missing input .gjf/.xyz file path".to_string())?;
    Ok(CliArgs {
        input_path,
        task,
        method,
        xc,
        basis,
        grid,
        rij, jfit_basis,
        direct,
        gpu,
        charge_override,
        mult_override,
        max_iter,
        conv_energy,
        conv_density,
        diis_max_vecs,
        solv_model,
        solvent,
        dispersion,
        tda, spin, threads, freq_step, freq_jobs, optimizer, checkpoint, resume,
        nstates,
        opt_max_cycles,
        json,
        out_path,
    })
}

fn parse_atom_token(tok: &str) -> Result<String, String> {
    let base = tok.split('(').next().unwrap_or(tok).trim();
    if let Ok(z) = base.parse::<u8>() {
        return symbol_from_atomic_number(z).map_err(|e| e.to_string());
    }
    Ok(base.to_string())
}

fn parse_atom_line(line: &str, in_bohr: bool) -> Result<Atom, String> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 4 {
        return Err(format!("Invalid coordinate line: '{line}'"));
    }
    let sym = parse_atom_token(parts[0])?;
    let x: f64 = parts[1].parse().map_err(|e| format!("Bad x in '{line}': {e}"))?;
    let y: f64 = parts[2].parse().map_err(|e| format!("Bad y in '{line}': {e}"))?;
    let z: f64 = parts[3].parse().map_err(|e| format!("Bad z in '{line}': {e}"))?;
    if in_bohr {
        Atom::from_bohr(&sym, [x, y, z]).map_err(|e| e.to_string())
    } else {
        Atom::from_angstrom(&sym, [x, y, z]).map_err(|e| e.to_string())
    }
}

fn load_molecule(path_str: &str) -> Result<Molecule, String> {
    let p = Path::new(path_str);
    if !p.exists() {
        return Err(format!("Input file not found: {path_str}"));
    }
    let content = std::fs::read_to_string(p).map_err(|e| format!("Failed to read '{path_str}': {e}"))?;
    let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("").to_ascii_lowercase();

    if ext == "xyz" {
        let mut lines = content.lines();
        let count = positive_count(lines.next().unwrap_or("").trim(), "XYZ atom count")?;
        lines.next().ok_or_else(|| "Missing XYZ comment line".to_string())?;
        let atoms = lines.filter(|l| !l.trim().is_empty()).map(|l| parse_atom_line(l, false)).collect::<Result<Vec<_>,_>>()?;
        if atoms.len() != count { return Err(format!("XYZ declared {count} atoms, parsed {}", atoms.len())); }
        return Ok(Molecule::new(atoms, 0, 1));
    }

    // Default: Gaussian .gjf / .com format
    let mut in_route = false;
    let mut in_bohr = false;
    let mut charge_mult: Option<(i32, usize)> = None;
    let mut atoms = Vec::new();
    let mut passed_blank_after_route = false;
    let mut title_seen = false;
    let mut post_title_blank = false;

    for raw in content.lines() {
        let line = raw.trim();
        if !passed_blank_after_route {
            if line.starts_with('#') {
                in_route = true;
                let lower = line.to_ascii_lowercase();
                if lower.contains("units=bohr") || lower.contains("units(bohr)") {
                    in_bohr = true;
                }
                continue;
            }
            if in_route {
                if line.is_empty() {
                    passed_blank_after_route = true;
                } else {
                    let lower = line.to_ascii_lowercase();
                    if lower.contains("units=bohr") || lower.contains("units(bohr)") {
                        in_bohr = true;
                    }
                }
                continue;
            }
            continue;
        }

        if !title_seen {
            if !line.is_empty() {
                title_seen = true;
            }
            continue;
        }

        if !post_title_blank {
            if line.is_empty() {
                post_title_blank = true;
            }
            continue;
        }

        if charge_mult.is_none() {
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let c: i32 = parts[0].parse().map_err(|e| format!("Bad charge in '{line}': {e}"))?;
                let m: usize = parts[1].parse().map_err(|e| format!("Bad mult in '{line}': {e}"))?;
                charge_mult = Some((c, m));
                continue;
            }
            return Err(format!("Invalid charge and multiplicity: {line}"));
        }

        if line.is_empty() {
            if !atoms.is_empty() {
                break;
            }
            continue;
        }

        if atoms.is_empty() && line.starts_with('@') {
            continue;
        }

        atoms.push(parse_atom_line(line, in_bohr)?);
    }

    if atoms.is_empty() {
        return Err(format!("No atom coordinates parsed from '{path_str}'"));
    }
    let (charge, mult) = charge_mult.ok_or("Missing Gaussian charge/multiplicity")?;
    Ok(Molecule::new(atoms, charge, mult))
}

fn parse_grid_level(s: &str) -> Result<GridLevel, String> {
    match s.to_ascii_lowercase().as_str() {
        "coarse" => Ok(GridLevel::Coarse),
        "medium" => Ok(GridLevel::Medium),
        "fine" => Ok(GridLevel::Fine),
        "ultrafine" => Ok(GridLevel::UltraFine),
        other => Err(format!("Unknown grid level '{other}'")),
    }
}

#[derive(Serialize)]
struct UnitsJson {
    energy: &'static str,
    coordinates: &'static str,
    gradient: &'static str,
    frequencies: &'static str,
    excitation_energy: &'static str,
    solvation_energy: &'static str,
}

#[derive(Serialize)]
struct JsonReport {
    status: &'static str,
    file: String,
    task: Task,
    method: Method,
    xc: String,
    basis: String,
    n_atoms: usize,
    n_ao: usize,
    threads: usize,
    initial_geometry_sha256: String,
    final_geometry_sha256: String,
    units: UnitsJson,
    timing_sec: TimingBreakdown,
    converged: bool,
    scf_iterations: usize,
    grid: Option<GridLevel>,
    direct_scf: bool,
    density_fitting: bool,
    gpu_requested: bool,
    scf: ScfSummary,
    task_settings: serde_json::Value,
    energy_hartree: f64,
    electronic_energy_hartree: f64,
    nuclear_repulsion_hartree: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    dispersion_energy_hartree: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gradient_atomic_units: Option<Vec<[f64; 3]>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_gradient_atomic_units: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rms_gradient_atomic_units: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    optimized_atoms: Option<Vec<OptimizedAtomJson>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    solvation_energy_kcal_mol: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequencies_cm1: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    zero_point_energy_hartree: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gibbs_free_energy_hartree: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    excited_states: Option<Vec<ExcitedStateJson>>,
}

#[derive(Serialize)]
struct TimingBreakdown {
    parse_sec: f64,
    basis_sec: f64,
    grid_sec: f64,
    scf_sec: f64,
    post_scf_sec: f64,
    total_wall_sec: f64,
}

#[derive(Serialize)]
struct OptimizedAtomJson {
    symbol: String,
    x_angstrom: f64,
    y_angstrom: f64,
    z_angstrom: f64,
}

#[derive(Serialize)]
struct ExcitedStateJson {
    label: String,
    residual_norm_hartree: Option<f64>,
    multiplicity: usize,
    dominant_transitions: Vec<calchem_tddft::OrbitalTransition>,
    state: usize,
    energy_ev: f64,
    wavelength_nm: f64,
    oscillator_strength: f64,
}

fn emit_output(report: &JsonReport, as_json: bool, out_file: Option<&str>) -> Result<(), String> {
    let out_str = if as_json {
        serde_json::to_string_pretty(report).map_err(|e| format!("JSON serialization error: {e}"))?
    } else {
        let mut s = String::new();
        s.push_str("\n======================================================================\n");
        s.push_str("                       CALCHEM QUANTUM RESULTS                        \n");
        s.push_str("======================================================================\n");
        s.push_str(&format!("Final Electronic Energy    : {:.12} Hartree\n", report.electronic_energy_hartree));
        s.push_str(&format!("Nuclear Repulsion Energy   : {:.12} Hartree\n", report.nuclear_repulsion_hartree));
        if let Some(edisp) = report.dispersion_energy_hartree {
            s.push_str(&format!("Empirical Dispersion (D3) : {:.12} Hartree\n", edisp));
        }
        if let Some(esolv) = report.solvation_energy_kcal_mol {
            s.push_str(&format!("Solvation Free Energy      : {:.4} kcal/mol\n", esolv));
        }
        s.push_str(&format!("TOTAL MOLECULAR ENERGY     : {:.12} Hartree\n", report.energy_hartree));
        s.push_str(&format!("SCF Convergence Status     : {} ({} iterations)\n", if report.converged { "CONVERGED" } else { "NOT CONVERGED" }, report.scf_iterations));

        if let (Some(max_g), Some(rms_g)) = (report.max_gradient_atomic_units, report.rms_gradient_atomic_units) {
            s.push_str("\n----------------------------------------------------------------------\n");
            s.push_str(&format!("Max Force: {:.6e} Eh/Bohr | RMS Force: {:.6e} Eh/Bohr\n", max_g, rms_g));
        }

        if let Some(ref states) = report.excited_states {
            s.push_str("\n----------------------------------------------------------------------\n");
            s.push_str("                     TD-DFT EXCITED STATE SPECTRUM                   \n");
            s.push_str("----------------------------------------------------------------------\n");
            s.push_str(" State     Energy (eV)     Wavelength (nm)     Oscillator Strength (f)\n");
            for st in states {
                s.push_str(&format!("  {:>3}       {:.4}           {:.2}                 {:.6}\n", st.label, st.energy_ev, st.wavelength_nm, st.oscillator_strength));
            }
        }

        if let Some(ref freqs) = report.frequencies_cm1 {
            s.push_str("\n----------------------------------------------------------------------\n");
            s.push_str("                     HARMONIC VIBRATIONAL FREQUENCIES                 \n");
            s.push_str("----------------------------------------------------------------------\n");
            for (i, f) in freqs.iter().enumerate() {
                s.push_str(&format!(" Mode {:03}: {:10.2} cm^-1\n", i + 1, f));
            }
            if let Some(zpe) = report.zero_point_energy_hartree {
                s.push_str(&format!("\nZero-Point Energy (ZPE)    : {:.6} Hartree\n", zpe));
            }
            if let Some(g) = report.gibbs_free_energy_hartree {
                s.push_str(&format!("Gibbs Free Energy G(298K)  : {:.6} Hartree\n", g));
            }
        }

        s.push_str("======================================================================\n");
        s.push_str(&format!("Total Wall-Clock Runtime   : {:.4} seconds\n", report.timing_sec.total_wall_sec));
        s.push_str("======================================================================\n");
        s
    };

    println!("{out_str}");
    if let Some(path) = out_file {
        let p = Path::new(path);
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("Failed to create output directory: {e}"))?;
            }
        }
        let tmp = p.with_file_name(format!(
            "{}.{}.tmp",
            p.file_name().unwrap_or_default().to_string_lossy(),
            std::process::id()
        ));
        std::fs::write(&tmp, out_str)
            .map_err(|e| format!("Failed to write temp output '{path}': {e}"))?;
        std::fs::rename(&tmp, p)
            .map_err(|e| format!("Failed to atomically move output to '{path}': {e}"))?;
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let cli = parse_args()?;
    if cli.out_path.as_deref().is_some_and(|out| Path::new(out) == Path::new(&cli.input_path)) {
        return Err("--out path must be separate from input file".into());
    }
    if cli.resume.is_some() && cli.task != Task::Opt {
        return Err("--resume requires --task opt".into());
    }
    if cli.checkpoint.is_some() && !matches!(cli.task, Task::Opt | Task::Freq) {
        return Err("--checkpoint requires --task opt or --task freq".into());
    }
    if let Some(n) = cli.threads {
        rayon::ThreadPoolBuilder::new().num_threads(n).build_global().map_err(|e| e.to_string())?;
    }
    let t_total_start = Instant::now();

    let t0 = Instant::now();
    let mut mol = load_molecule(&cli.input_path)?;
    if let Some(c) = cli.charge_override {
        mol.charge = c;
    }
    if let Some(m) = cli.mult_override {
        mol.multiplicity = m;
    }
    let initial_geom_hash = checkpoint::geometry_sha256(&mol);
    let mut final_geom_hash = initial_geom_hash.clone();
    let t_parse = t0.elapsed();

    let t1 = Instant::now();
    let basis = BasisSet::from_name(&cli.basis, &mol).map_err(|e| e.to_string())?;
    let t_basis = t1.elapsed();

    let opts = ScfOptions {
        max_iter: cli.max_iter,
        conv_energy: cli.conv_energy,
        conv_density: cli.conv_density,
        use_diis: true,
        diis_max_vecs: cli.diis_max_vecs,
        level_shift: 0.0,
        damping: 0.0,
        direct: cli.direct,
        density_fitting: cli.rij,
        use_gpu: cli.gpu,
        diis_disk: false,
        diis_dir: None,
        jfit_basis: cli.jfit_basis,
    };

    let threads = rayon::current_num_threads();
    eprintln!(
        "CalChem | task={:?} file={} atoms={} charge={} mult={} method={:?} xc={} basis={} n_ao={} n_shells={} threads={}",
        cli.task,
        cli.input_path,
        mol.atoms.len(),
        mol.charge,
        mol.multiplicity,
        cli.method,
        cli.xc,
        cli.basis,
        basis.n_ao,
        basis.shells.len(),
        threads
    );

    let xc = XcFunctional::from_name(&cli.xc).map_err(|e| e.to_string())?;
    let grid_level = parse_grid_level(&cli.grid)?;

    let reference = match cli.method {
        Method::Rhf => Reference::Rhf, Method::Uhf => Reference::Uhf,
        Method::Rks => Reference::Rks(xc), Method::Uks => Reference::Uks(xc),
    };
    let is_dft = matches!(cli.method, Method::Rks | Method::Uks);
    let functional = if is_dft { xc.canonical_name() } else { "hf" };
    if is_dft && xc == XcFunctional::Wb97xd && cli.dispersion.is_some() {
        return Err("wB97X-D already includes built-in D2 dispersion; do not combine with --disp".into());
    }
    let d3 = cli.dispersion.as_deref().map(|scheme| D3Model::from_scheme(scheme, functional)).transpose().map_err(|e|e.to_string())?;
    let solv = match cli.solv_model.as_deref() {
        Some("pcm" | "iefpcm") => Some(calchem_solv::SolventModel::IefPcm),
        Some("cpcm") => Some(calchem_solv::SolventModel::Cpcm),
        Some("smd") => Some(calchem_solv::SolventModel::Smd),
        Some(v) => return Err(format!("Unknown --solv model '{v}'")),
        None if cli.solvent.is_some() => return Err("--solvent requires --solv".into()),
        None => None,
    };
    let resolved_solvent = if solv.is_some() {
        let name = cli.solvent.as_deref().unwrap_or("water");
        let s = calchem_solv::Solvent::from_name(name)
            .ok_or_else(|| format!("Unknown solvent '{name}'"))?;
        Some((name.to_string(), s))
    } else {
        None
    };
    if solv.is_some() && cli.task != Task::Energy { return Err("CLI continuum solvation (SCRF) is supported for single-point energy only; derivatives and excited-state response require consistent solvent derivatives/response".into()); }
    if cli.task == Task::TdDft && (cli.rij || matches!(cli.method,Method::Uhf|Method::Uks)) {
        return Err("TD response requires a restricted exact-Coulomb reference".into());
    }
    let mut scf_time = std::time::Duration::ZERO;
    let scf;
    let mut task_converged = true;
    let t_post0 = Instant::now();
    let mut settings = serde_json::json!({});
    let mut grad_out = None;
    let mut max_g_val = None;
    let mut rms_g_val = None;
    let mut opt_atoms_out = None;
    let mut solv_energy_out = None;
    let mut freqs_out = None;
    let mut zpe_out = None;
    let mut gibbs_out = None;
    let mut excited_states_out = None;
    let final_energy;
    let e_disp;
    match cli.task {
        Task::Energy if solv.is_some() => {
            let (Some(model), Some((name, solvent))) = (solv, resolved_solvent) else {
                unreachable!("solv.is_some() guarantees resolved_solvent");
            };
            let method_str = match cli.method {
                Method::Rhf => "rhf",
                Method::Uhf => "uhf",
                Method::Rks => "rks",
                Method::Uks => "uks",
            };
            let solv_opts = calchem_solv::SolvationOptions {
                solvent,
                model,
                ..Default::default()
            };
            let start = Instant::now();
            let res = calchem_solv::run_scrf_with_grid(
                &mol,
                &basis,
                &opts,
                &solv_opts,
                method_str,
                is_dft.then_some(xc),
                grid_level,
            )
            .map_err(|e| e.to_string())?;
            scf_time = start.elapsed();
            scf = (&res.wavefunction).into();
            task_converged = res.converged;
            e_disp = d3.as_ref().map(|d| d.energy(&mol.atoms)).transpose().map_err(|e| e.to_string())?;
            let disp_val = e_disp.unwrap_or(0.0);
            final_energy = res.solvated_energy + disp_val;
            solv_energy_out = Some(res.delta_g_solv_kcal_mol);
            settings = serde_json::json!({
                "solvation": "scrf",
                "model": model,
                "solvent": name,
                "gas_energy_hartree": res.gas_energy + disp_val,
                "solvated_energy_hartree": final_energy,
                "delta_g_solv_kcal_mol": res.delta_g_solv_kcal_mol,
                "electrostatic_kcal_mol": res.delta_g_elec_kcal_mol,
                "cds_kcal_mol": res.delta_g_cds_kcal_mol,
                "cavity_surface_area_angstrom2": res.cavity_surface_area_angstrom2,
                "total_surface_charge": res.total_surface_charge,
            });
        }
        Task::Energy | Task::Gradient | Task::TdDft => {
            let start = Instant::now();
            let point = GasPoint::solve(&mol, &basis, reference, grid_level, &opts).map_err(|e|e.to_string())?;
            scf_time = start.elapsed();
            scf = point.summary();
            e_disp = d3.as_ref().map(|d| d.energy(&mol.atoms)).transpose().map_err(|e|e.to_string())?;
            final_energy = point.energy() + e_disp.unwrap_or(0.0);
            if cli.task == Task::Gradient {
                let mut g = point.gradient(&mol,&basis).map_err(|e|e.to_string())?;
                if let Some(d) = &d3 { d.add_gradient(&mol.atoms,&mut g).map_err(|e|e.to_string())?; }
            let mut max_g: f64 = 0.0;
            let mut sum_g2: f64 = 0.0;
            for atom_g in &g {
                for &comp in atom_g {
                    let a = comp.abs();
                    if a > max_g { max_g = a; }
                    sum_g2 += comp * comp;
                }
            }
            let rms_g = (sum_g2 / (3.0 * mol.atoms.len() as f64)).sqrt();
            max_g_val = Some(max_g);
            rms_g_val = Some(rms_g);
            grad_out = Some(g);
            }
            if cli.task == Task::TdDft {
            let td_opts = calchem_tddft::TddftOptions {
                n_states: cli.nstates,
                tda: cli.tda,
                manifold: cli.spin,
                grid_level,
                scf_options: opts.clone(),
                ..Default::default()
            };
            let GasPoint::Restricted(wfn,_) = &point else { return Err("TD requires restricted orbitals".into()); };
            let td_res = calchem_tddft::run_tddft_from_scf(&mol, &basis, wfn, is_dft.then_some(xc), &td_opts).map_err(|e| e.to_string())?;
            let required = if cli.spin == SpinManifold::Both { 2 } else { 1 };
            if td_res.states.len() / required != cli.nstates { return Err(format!("Requested {} roots per manifold, obtained {} total roots",cli.nstates,td_res.states.len())); }
            settings = serde_json::json!({"td_mode": if cli.tda {"tda"} else {"lr"}, "manifold":cli.spin, "roots_per_manifold":cli.nstates,"occupied":td_res.n_occ,"virtual":td_res.n_virt,"singlet_iterations":td_res.davidson_iterations_singlet,"triplet_iterations":td_res.davidson_iterations_triplet});
            let mut states = Vec::new();
            for state in &td_res.states {
                states.push(ExcitedStateJson {
                    state: state.state_index, label: state.label.clone(), residual_norm_hartree: state.residual_norm_hartree, multiplicity: state.multiplicity, dominant_transitions: state.dominant_transitions.clone(),
                    energy_ev: state.excitation_energy_ev,
                    wavelength_nm: state.wavelength_nm,
                    oscillator_strength: state.oscillator_strength,
                });
            }
            excited_states_out = Some(states);
            }
        }
        Task::Opt => {
            let opt_opts = calchem_opt::OptOptions {
                max_steps: cli.opt_max_cycles,
                reference, grid_level, dispersion: cli.dispersion.clone(),
                scf_options: opts.clone(), ..cli.optimizer.clone()
            };
            let mut progress = checkpoint::Progress::new(&cli, &mol, &opt_opts)?;
            let resume = progress.load(cli.resume.as_deref())?;
            let opt_res = match calchem_opt::optimize_geometry_observed(&mol, &cli.basis, &opt_opts, resume, &mut progress) {
                Ok(r) => r,
                Err(e) => {
                    if let (Some(cp), Some((step, energy, max_f))) = (progress.path(), progress.last_saved()) {
                        eprintln!(
                            "opt partial checkpoint preserved at {} (last_step={step} E={energy:.12} max_force={max_f:.4e})",
                            cp.display()
                        );
                    }
                    return Err(e.to_string());
                }
            };
            task_converged = opt_res.converged;
            final_geom_hash = checkpoint::geometry_sha256(&opt_res.molecule);
            scf = opt_res.final_scf.ok_or("Missing final SCF diagnostics")?;
            e_disp = d3.as_ref().map(|d| d.energy(&opt_res.molecule.atoms)).transpose().map_err(|e|e.to_string())?;
            settings = serde_json::json!({"options":opt_opts,"steps":opt_res.steps,"history":opt_res.history,"checkpoint":progress.path().map(|p| p.display().to_string())});
            if let Some(last) = opt_res.history.last() { max_g_val = Some(last.max_force); rms_g_val = Some(last.rms_force); }
            grad_out = Some(opt_res.gradient);
            final_energy = opt_res.energy_hartree;
            let mut atoms_list = Vec::new();
            for a in &opt_res.molecule.atoms {
                let x = a.position_bohr[0] * calchem_model::BOHR_TO_ANGSTROM;
                let y = a.position_bohr[1] * calchem_model::BOHR_TO_ANGSTROM;
                let z = a.position_bohr[2] * calchem_model::BOHR_TO_ANGSTROM;
                atoms_list.push(OptimizedAtomJson {
                    symbol: a.symbol.clone(),
                    x_angstrom: x,
                    y_angstrom: y,
                    z_angstrom: z,
                });
            }
            opt_atoms_out = Some(atoms_list);
        }
        Task::Freq => {
            let freq_opts = calchem_vib::FreqOptions {
                step_bohr: cli.freq_step,
                grid_level,
                temperature_k: 298.15,
                pressure_pa: 101325.0,
                symmetry_number: None,
                scf_options: opts.clone(),
            };
            let vib_method = match cli.method {
                Method::Rks => calchem_vib::VibMethod::Rks(xc),
                Method::Rhf => calchem_vib::VibMethod::Rhf,
                Method::Uks => calchem_vib::VibMethod::Uks(xc),
                Method::Uhf => calchem_vib::VibMethod::Uhf,
            };
            let freq_cp_dir = cli.checkpoint.as_deref().map(Path::new);
            let freq_fp = if freq_cp_dir.is_some() {
                Some(checkpoint::freq_fingerprint(&cli, &mol, &freq_opts)?)
            } else {
                None
            };
            let (f_res, center, correction) = calchem_vib::compute_frequencies_with_checkpoint(
                &mol,
                &cli.basis,
                vib_method,
                &freq_opts,
                d3.as_ref(),
                cli.freq_jobs,
                freq_cp_dir,
                freq_fp.as_deref(),
            )
            .map_err(|e| e.to_string())?;
            final_energy = f_res.energy_hartree;
            scf = center;
            e_disp = d3.as_ref().map(|_|correction);
            settings = serde_json::json!({"step_bohr":cli.freq_step,"displacement_jobs":cli.freq_jobs,"displaced_gradients":6 * mol.atoms.len(),"checkpoint_dir":cli.checkpoint});
            freqs_out = Some(f_res.vibrational_analysis.modes.iter().map(|m| m.frequency_cm).collect());
            zpe_out = Some(f_res.thermochemistry.zpve_hartree);
            gibbs_out = Some(f_res.thermochemistry.gibbs_free_energy_hartree);
        }

    }
    let t_post = t_post0.elapsed();
    let all_converged = scf.converged && task_converged;

    let report = JsonReport {
        status: if all_converged { "converged" } else { "not_converged" },
        file: cli.input_path.clone(),
        task: cli.task,
        method: cli.method,
        xc: functional.to_string(),
        basis: cli.basis.clone(),
        n_atoms: mol.atoms.len(),
        n_ao: basis.n_ao,
        threads,
        initial_geometry_sha256: initial_geom_hash,
        final_geometry_sha256: final_geom_hash,
        units: UnitsJson {
            energy: "hartree",
            coordinates: "angstrom",
            gradient: "hartree/bohr",
            frequencies: "cm^-1",
            excitation_energy: "eV",
            solvation_energy: "kcal/mol",
        },
        timing_sec: TimingBreakdown {
            parse_sec: t_parse.as_secs_f64(),
            basis_sec: t_basis.as_secs_f64(),
            grid_sec: 0.0,
            scf_sec: scf_time.as_secs_f64(),
            post_scf_sec: t_post.saturating_sub(scf_time).as_secs_f64(),
            total_wall_sec: t_total_start.elapsed().as_secs_f64(),
        },
        converged: all_converged,
        scf_iterations: scf.iterations,
        energy_hartree: final_energy,
        electronic_energy_hartree: scf.electronic_energy_hartree,
        nuclear_repulsion_hartree: scf.nuclear_repulsion_hartree,
        dispersion_energy_hartree: e_disp,
        grid: is_dft.then_some(grid_level), direct_scf: calchem_scf::should_use_direct(&basis,&opts), density_fitting: opts.density_fitting, gpu_requested: opts.use_gpu,
        scf, task_settings: settings,
        gradient_atomic_units: grad_out,
        max_gradient_atomic_units: max_g_val,
        rms_gradient_atomic_units: rms_g_val,
        optimized_atoms: opt_atoms_out,
        solvation_energy_kcal_mol: solv_energy_out,
        frequencies_cm1: freqs_out,
        zero_point_energy_hartree: zpe_out,
        gibbs_free_energy_hartree: gibbs_out,
        excited_states: excited_states_out,
    };

    emit_output(&report, cli.json, cli.out_path.as_deref())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
