use calchem_basis::BasisSet;
use calchem_dft::{GridLevel, MolecularGrid, XcFunctional};
use calchem_model::{symbol_from_atomic_number, Atom, Molecule, ScfOptions, ScfWavefunction};
use calchem_scf::{rhf, rks_with_grid, uhf, uks_with_grid};
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
         \t--solv <pcm|smd>                     Implicit continuum solvation model\n\
         \t--solvent <name>                     Solvent name (e.g. water, chloroform)\n\
         \t--disp <d3bj>                        Grimme D3(BJ) dispersion correction\n\
         \t--nstates <uint>                     Excited states count for TD-DFT (default: 3)\n\
         \t--opt-max-cycles <uint>              Max geometry optimization steps (default: 50)\n\n\
         Output Formats:\n\
         \t--json                               Emit structured machine-readable JSON\n\
         \t--out <file>                         Write output to specified file\n\
         \t-h, --help                           Show this help message"
    );
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
                max_iter = v.parse::<usize>().map_err(|e| format!("Invalid --max-iter '{v}': {e}"))?;
            }
            "--conv-e" => {
                let v = next_val(&mut i, "--conv-e")?;
                conv_energy = v.parse::<f64>().map_err(|e| format!("Invalid --conv-e '{v}': {e}"))?;
            }
            "--conv-d" => {
                let v = next_val(&mut i, "--conv-d")?;
                conv_density = v.parse::<f64>().map_err(|e| format!("Invalid --conv-d '{v}': {e}"))?;
            }
            "--diis-max-vecs" => {
                let v = next_val(&mut i, "--diis-max-vecs")?;
                diis_max_vecs = v.parse::<usize>().map_err(|e| format!("Invalid --diis-max-vecs '{v}': {e}"))?;
            }
            "--solv" => solv_model = Some(next_val(&mut i, "--solv")?.to_ascii_lowercase()),
            "--solvent" => solvent = Some(next_val(&mut i, "--solvent")?.to_ascii_lowercase()),
            "--disp" => dispersion = Some(next_val(&mut i, "--disp")?.to_ascii_lowercase()),
            "--nstates" => {
                let v = next_val(&mut i, "--nstates")?;
                nstates = v.parse::<usize>().map_err(|e| format!("Invalid --nstates '{v}': {e}"))?;
            }
            "--opt-max-cycles" => {
                let v = next_val(&mut i, "--opt-max-cycles")?;
                opt_max_cycles = v.parse::<usize>().map_err(|e| format!("Invalid --opt-max-cycles '{v}': {e}"))?;
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
        rij,
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
        let mut lines = content.lines().map(|s| s.trim()).filter(|s| !s.is_empty());
        let _natoms_line = lines.next().ok_or_else(|| "Empty XYZ file".to_string())?;
        let _comment = lines.next();
        let mut atoms = Vec::new();
        for line in lines {
            atoms.push(parse_atom_line(line, false)?);
        }
        if atoms.is_empty() {
            return Err("No atoms found in XYZ file".to_string());
        }
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

        if let Ok(atom) = parse_atom_line(line, in_bohr) {
            atoms.push(atom);
        }
    }

    if atoms.is_empty() {
        return Err(format!("No atom coordinates parsed from '{path_str}'"));
    }
    let (charge, mult) = charge_mult.unwrap_or((0, 1));
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
struct JsonReport {
    file: String,
    task: Task,
    method: Method,
    xc: String,
    basis: String,
    n_atoms: usize,
    n_ao: usize,
    threads: usize,
    timing_sec: TimingBreakdown,
    converged: bool,
    scf_iterations: usize,
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
                s.push_str(&format!("  S{:02}       {:.4}           {:.2}                 {:.6}\n", st.state, st.energy_ev, st.wavelength_nm, st.oscillator_strength));
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
        std::fs::write(path, out_str).map_err(|e| format!("Failed to write output to '{path}': {e}"))?;
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let cli = parse_args()?;
    let t_total_start = Instant::now();

    let t0 = Instant::now();
    let mut mol = load_molecule(&cli.input_path)?;
    if let Some(c) = cli.charge_override {
        mol.charge = c;
    }
    if let Some(m) = cli.mult_override {
        mol.multiplicity = m;
    }
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
    };

    let threads = rayon::current_num_threads();
    println!(
        "CalChem Native Industrial Engine | task={:?} file={} atoms={} charge={} mult={} method={:?} xc={} basis={} n_ao={} n_shells={} threads={}",
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

    let t2 = Instant::now();
    let grid = MolecularGrid::build_for_basis(&mol, &basis, grid_level);
    let t_grid = t2.elapsed();

    // 1. Core SCF Execution
    let t3 = Instant::now();
    let wfn: ScfWavefunction = match cli.method {
        Method::Rks => rks_with_grid(&mol, &basis, xc, &grid, &opts).map_err(|e| e.to_string())?,
        Method::Rhf => rhf(&mol, &basis, &opts).map_err(|e| e.to_string())?,
        Method::Uks => {
            let u_wfn = uks_with_grid(&mol, &basis, xc, &grid, &opts).map_err(|e| e.to_string())?;
            let mut occ = vec![0.0; u_wfn.n_ao];
            for i in 0..u_wfn.n_alpha { occ[i] += 1.0; }
            for i in 0..u_wfn.n_beta { occ[i] += 1.0; }
            ScfWavefunction {
                density_fitting: u_wfn.density_fitting,
                converged: u_wfn.converged,
                iterations: u_wfn.iterations,
                total_energy_hartree: u_wfn.total_energy_hartree,
                electronic_energy_hartree: u_wfn.electronic_energy_hartree,
                nuclear_repulsion_hartree: u_wfn.nuclear_repulsion_hartree,
                n_ao: u_wfn.n_ao,
                n_mo: u_wfn.n_ao,
                mo_energies: u_wfn.mo_energies_alpha.clone(),
                mo_occupations: occ,
                mo_coefficients: u_wfn.mo_coefficients_alpha.clone(),
                density_alpha: u_wfn.density_alpha.clone(),
                density_beta: Some(u_wfn.density_beta.clone()),
                fock_alpha: vec![0.0; u_wfn.n_ao * u_wfn.n_ao],
                overlap_matrix: u_wfn.overlap_matrix.clone(),
                h_core_matrix: vec![0.0; u_wfn.n_ao * u_wfn.n_ao],
                diagnostics: u_wfn.diagnostics.clone(),
            }
        }
        Method::Uhf => {
            let u_wfn = uhf(&mol, &basis, &opts).map_err(|e| e.to_string())?;
            let mut occ = vec![0.0; u_wfn.n_ao];
            for i in 0..u_wfn.n_alpha { occ[i] += 1.0; }
            for i in 0..u_wfn.n_beta { occ[i] += 1.0; }
            ScfWavefunction {
                density_fitting: u_wfn.density_fitting,
                converged: u_wfn.converged,
                iterations: u_wfn.iterations,
                total_energy_hartree: u_wfn.total_energy_hartree,
                electronic_energy_hartree: u_wfn.electronic_energy_hartree,
                nuclear_repulsion_hartree: u_wfn.nuclear_repulsion_hartree,
                n_ao: u_wfn.n_ao,
                n_mo: u_wfn.n_ao,
                mo_energies: u_wfn.mo_energies_alpha.clone(),
                mo_occupations: occ,
                mo_coefficients: u_wfn.mo_coefficients_alpha.clone(),
                density_alpha: u_wfn.density_alpha.clone(),
                density_beta: Some(u_wfn.density_beta.clone()),
                fock_alpha: vec![0.0; u_wfn.n_ao * u_wfn.n_ao],
                overlap_matrix: u_wfn.overlap_matrix.clone(),
                h_core_matrix: vec![0.0; u_wfn.n_ao * u_wfn.n_ao],
                diagnostics: u_wfn.diagnostics.clone(),
            }
        }
    };
    let t_scf = t3.elapsed();

    // 2. Post-SCF Task Execution
    let t_post0 = Instant::now();
    let mut grad_out = None;
    let mut max_g_val = None;
    let mut rms_g_val = None;
    let mut opt_atoms_out = None;
    let mut solv_energy_out = None;
    let mut freqs_out = None;
    let mut zpe_out = None;
    let mut gibbs_out = None;
    let mut excited_states_out = None;
    let mut final_energy = wfn.total_energy_hartree;

    // Optional dispersion
    let mut e_disp = 0.0;
    if cli.dispersion.as_deref() == Some("d3bj") {
        if let Ok(d3_model) = calchem_dft::D3Model::d3bj(xc.canonical_name()) {
            if let Ok((e_d3, _)) = d3_model.energy_and_gradient(&mol.atoms, true) {
                e_disp = e_d3;
                final_energy += e_disp;
            }
        }
    }

    // Optional Solvation
    if let Some(ref s_model) = cli.solv_model {
        if s_model == "pcm" || s_model == "smd" {
            let solv_name = cli.solvent.as_deref().unwrap_or("water");
            let s_obj = calchem_solv::Solvent::from_name(solv_name)
                .ok_or_else(|| format!("Unknown solvent: {solv_name}"))?;
            let s_opts = calchem_solv::SolvationOptions {
                solvent: s_obj,
                model: if s_model == "smd" { calchem_solv::SolventModel::Smd } else { calchem_solv::SolventModel::Cpcm },
                ..Default::default()
            };
            let s_res = calchem_solv::compute_pcm_solvation(&mol, &basis, &wfn, &s_opts).map_err(|e| e.to_string())?;
            solv_energy_out = Some(s_res.delta_g_solv_kcal_mol);
            final_energy += s_res.delta_g_solv;
        }
    }

    match cli.task {
        Task::Energy => {}
        Task::Gradient => {
            let g = match cli.method {
                Method::Rks => calchem_grad::rks_gradient_with_grid(&mol, &basis, xc, &grid, &wfn).map_err(|e| e.to_string())?,
                Method::Rhf => calchem_grad::rhf_gradient(&mol, &basis, &wfn).map_err(|e| e.to_string())?,
                _ => return Err(format!("Gradient currently implemented for RKS and RHF")),
            };
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
        Task::Opt => {
            let opt_opts = calchem_opt::OptOptions {
                max_steps: cli.opt_max_cycles,
                reference: calchem_opt::Reference::from_method_and_xc(
                    match cli.method { Method::Rks => "rks", Method::Rhf => "rhf", Method::Uks => "uks", Method::Uhf => "uhf" },
                    Some(&cli.xc)
                ).map_err(|e| e.to_string())?,
                grid_level,
                scf_options: opts.clone(),
                ..Default::default()
            };
            let opt_res = calchem_opt::optimize_geometry_with_options(&mol, &cli.basis, &opt_opts).map_err(|e| e.to_string())?;
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
                step_bohr: 0.001,
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
            let f_res = calchem_vib::compute_frequencies_and_thermo(&mol, &cli.basis, vib_method, &freq_opts).map_err(|e| e.to_string())?;
            freqs_out = Some(f_res.vibrational_analysis.modes.iter().map(|m| m.frequency_cm).collect());
            zpe_out = Some(f_res.thermochemistry.zpve_hartree);
            gibbs_out = Some(f_res.thermochemistry.gibbs_free_energy_hartree);
        }
        Task::TdDft => {
            let td_opts = calchem_tddft::TddftOptions {
                n_states: cli.nstates,
                tda: true,
                manifold: calchem_tddft::SpinManifold::Singlet,
                grid_level,
                scf_options: opts.clone(),
                ..Default::default()
            };
            let td_res = calchem_tddft::run_tddft_from_scf(&mol, &basis, &wfn, Some(xc), &td_opts).map_err(|e| e.to_string())?;
            let mut states = Vec::new();
            for state in &td_res.states {
                states.push(ExcitedStateJson {
                    state: state.state_index,
                    energy_ev: state.excitation_energy_ev,
                    wavelength_nm: state.wavelength_nm,
                    oscillator_strength: state.oscillator_strength,
                });
            }
            excited_states_out = Some(states);
        }
    }
    let t_post = t_post0.elapsed();

    let report = JsonReport {
        file: cli.input_path.clone(),
        task: cli.task,
        method: cli.method,
        xc: cli.xc.clone(),
        basis: cli.basis.clone(),
        n_atoms: mol.atoms.len(),
        n_ao: basis.n_ao,
        threads,
        timing_sec: TimingBreakdown {
            parse_sec: t_parse.as_secs_f64(),
            basis_sec: t_basis.as_secs_f64(),
            grid_sec: t_grid.as_secs_f64(),
            scf_sec: t_scf.as_secs_f64(),
            post_scf_sec: t_post.as_secs_f64(),
            total_wall_sec: t_total_start.elapsed().as_secs_f64(),
        },
        converged: wfn.converged,
        scf_iterations: wfn.iterations,
        energy_hartree: final_energy,
        electronic_energy_hartree: wfn.electronic_energy_hartree,
        nuclear_repulsion_hartree: wfn.nuclear_repulsion_hartree,
        dispersion_energy_hartree: if e_disp != 0.0 { Some(e_disp) } else { None },
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
