use super::CliArgs;
use calchem_model::{CalcError, Molecule, ScfSummary};
use calchem_opt::{OptCheckpoint, OptObserver, OptOptions, OptStep};
use calchem_tensor::Matrix2D;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Deterministic bit-level SHA-256 fingerprint of molecular geometry, charge, and multiplicity.
pub fn geometry_sha256(mol: &Molecule) -> String {
    let mut hash = Sha256::new();
    hash.update(mol.charge.to_le_bytes());
    hash.update((mol.multiplicity as u64).to_le_bytes());
    hash.update((mol.atoms.len() as u64).to_le_bytes());
    for a in &mol.atoms {
        hash.update(a.symbol.as_bytes());
        hash.update([0u8, a.z]);
        for &c in &a.position_bohr {
            hash.update(c.to_bits().to_le_bytes());
        }
    }
    format!("{:x}", hash.finalize())
}

fn hash_env_overrides(hash: &mut Sha256) -> Result<(), String> {
    for key in [
        "CALCHEM_AUTO_AUX_MAX_L",
        "CALCHEM_DIRECT_THRESH",
        "CALCHEM_AUX_BASIS",
        "CALCHEM_AUX_PROFILE",
        "CALCHEM_STREAM_GRID",
    ] {
        hash.update(serde_json::to_vec(&(key, std::env::var(key).ok())).map_err(|e| e.to_string())?);
    }
    Ok(())
}

pub(super) fn freq_fingerprint(
    cli: &CliArgs,
    mol: &Molecule,
    freq_opts: &calchem_vib::FreqOptions,
) -> Result<String, String> {
    let mut hash = Sha256::new();
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    hash.update(fs::read(exe).map_err(|e| e.to_string())?);
    hash.update(
        serde_json::to_vec(&(
            mol,
            &cli.basis,
            &cli.method,
            &cli.xc,
            &cli.dispersion,
            freq_opts,
        ))
        .map_err(|e| e.to_string())?,
    );
    hash_env_overrides(&mut hash)?;
    Ok(format!("{:x}", hash.finalize()))
}

#[derive(Serialize, Deserialize)]
struct FileCheckpoint {
    model: String,
    state: OptCheckpoint,
}

pub struct Progress {
    path: Option<PathBuf>,
    model: String,
    start_time: Instant,
    last_saved: Option<(usize, f64, f64)>,
}

impl Progress {
    pub(super) fn new(cli: &CliArgs, mol: &Molecule, options: &OptOptions) -> Result<Self, String> {
        let path = cli
            .checkpoint
            .clone()
            .or_else(|| cli.resume.clone())
            .or_else(|| cli.out_path.as_ref().map(|p| format!("{p}.checkpoint.json")))
            .map(PathBuf::from);
        if path.as_deref().is_some_and(|p| {
            cli.out_path
                .as_deref()
                .is_some_and(|out| p == Path::new(out))
                || p == Path::new(&cli.input_path)
        }) {
            return Err("Checkpoint path must be separate from input and output".into());
        }
        let mut config = options.clone();
        config.max_steps = 0; // A restart may extend the step budget.
        let mut hash = Sha256::new();
        if path.is_some() {
            let exe = std::env::current_exe().map_err(|e| e.to_string())?;
            hash.update(fs::read(exe).map_err(|e| e.to_string())?);
            hash.update(serde_json::to_vec(&(mol, &cli.basis, &config)).map_err(|e| e.to_string())?);
            hash_env_overrides(&mut hash)?;
        }
        Ok(Self {
            path,
            model: format!("{:x}", hash.finalize()),
            start_time: Instant::now(),
            last_saved: None,
        })
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    pub fn last_saved(&self) -> Option<(usize, f64, f64)> {
        self.last_saved
    }

    pub fn load(&self, path: Option<&str>) -> Result<Option<OptCheckpoint>, String> {
        let Some(p) = path else { return Ok(None) };
        let f = fs::File::open(p).map_err(|e| format!("Cannot open checkpoint {p}: {e}"))?;
        let checkpoint: FileCheckpoint = serde_json::from_reader(BufReader::new(f))
            .map_err(|e| format!("Invalid checkpoint {p}: {e}"))?;
        if self.model != checkpoint.model {
            return Err("Checkpoint binary, initial molecule or calculation settings differ".into());
        }
        Ok(Some(checkpoint.state))
    }
}

impl OptObserver for Progress {
    fn step(
        &mut self,
        mol: &Molecule,
        h: &Matrix2D,
        history: &[OptStep],
        gradient: &[[f64; 3]],
        scf: Option<&ScfSummary>,
    ) -> Result<(), CalcError> {
        let last = history
            .last()
            .ok_or_else(|| CalcError::Internal("Missing optimizer history".into()))?;
        let elapsed = self.start_time.elapsed().as_secs_f64();
        eprintln!(
            "opt step={} E={:.12} max_force={:.4e} rms_force={:.4e} displacement={:.4e} scf_iter={} trials={} alpha={:.4e} elapsed={:.2}s",
            last.step,
            last.energy_hartree,
            last.max_force,
            last.rms_force,
            last.max_displacement,
            last.scf_iterations,
            last.line_search_trials,
            last.step_alpha,
            elapsed
        );
        self.last_saved = Some((last.step, last.energy_hartree, last.max_force));
        let Some(p) = &self.path else { return Ok(()) };
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|e| {
                    CalcError::Internal(format!(
                        "Cannot create checkpoint dir {}: {e}",
                        parent.display()
                    ))
                })?;
            }
        }
        let state = OptCheckpoint {
            molecule: mol.clone(),
            inverse_hessian: h.data.clone(),
            history: history.to_vec(),
            gradient: gradient.to_vec(),
            final_scf: scf.cloned(),
        };
        let tmp = p.with_file_name(format!(
            "{}.{}.tmp",
            p.file_name().unwrap_or_default().to_string_lossy(),
            std::process::id()
        ));
        let save = || -> std::io::Result<()> {
            let file = fs::File::create(&tmp)?;
            let mut writer = BufWriter::new(file);
            serde_json::to_writer(
                &mut writer,
                &FileCheckpoint {
                    model: self.model.clone(),
                    state,
                },
            )?;
            writer.flush()?;
            writer.get_ref().sync_all()?;
            drop(writer);
            fs::rename(&tmp, p)
        };
        save().map_err(|e| {
            CalcError::Internal(format!("Cannot write checkpoint {}: {e}", p.display()))
        })
    }

    fn trial(
        &mut self,
        step: usize,
        attempt: usize,
        alpha: f64,
        result: Result<f64, &CalcError>,
        accepted: bool,
    ) {
        match result {
            Ok(energy) => eprintln!(
                "opt step={step} trial={attempt} alpha={alpha:.4e} E={energy:.12} accepted={accepted}"
            ),
            Err(e) => eprintln!("opt step={step} trial={attempt} alpha={alpha:.4e} error={e}"),
        }
    }
}
