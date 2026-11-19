use calchem_tensor::{solve_linear_system, Matrix2D};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

static DIIS_FILE_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
pub struct DiskDiisStore {
    path: PathBuf,
    file: File,
    rows: usize,
    cols: usize,
    max_vecs: usize,
    slots: Vec<usize>,
    dots: Vec<Vec<f64>>,
}

impl Drop for DiskDiisStore {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

impl DiskDiisStore {
    fn resolve_spill_dir() -> PathBuf {
        for var in ["CALCHEM_DIIS_DIR", "CALCHEM_TMPDIR"] {
            if let Ok(val) = std::env::var(var) {
                let trimmed = val.trim();
                if !trimmed.is_empty() {
                    return PathBuf::from(trimmed);
                }
            }
        }
        if let Ok(val) = std::env::var("CALCHEM_DATA") {
            let trimmed = val.trim();
            if !trimmed.is_empty() {
                return PathBuf::from(trimmed).join("diis_spill");
            }
        }
        let repo_runtime = PathBuf::from(r"E:\YDW\Trae_Gaussian\build_target\runtime\diis_spill");
        if PathBuf::from(r"E:\YDW\Trae_Gaussian").exists() {
            return repo_runtime;
        }
        std::env::temp_dir()
    }

    fn create(max_vecs: usize, rows: usize, cols: usize) -> std::io::Result<Self> {
        let dir = Self::resolve_spill_dir();
        Self::create_in_dir(&dir, max_vecs, rows, cols)
    }

    fn create_in_dir(
        dir: &std::path::Path,
        max_vecs: usize,
        rows: usize,
        cols: usize,
    ) -> std::io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        let id = DIIS_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let path = dir.join(format!("calchem_diis_{pid}_{id}.bin"));
        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(&path)?;
        Ok(Self {
            path,
            file,
            rows,
            cols,
            max_vecs: max_vecs.max(2),
            slots: Vec::new(),
            dots: Vec::new(),
        })
    }

    fn n_elems(&self) -> usize {
        self.rows * self.cols
    }

    fn slot_offset(&self, slot: usize, is_err: bool) -> u64 {
        let vec_bytes = (self.n_elems() * std::mem::size_of::<f64>()) as u64;
        let base = (slot as u64) * 2 * vec_bytes;
        if is_err {
            base + vec_bytes
        } else {
            base
        }
    }

    fn write_vec(&mut self, slot: usize, is_err: bool, data: &[f64]) -> std::io::Result<()> {
        let offset = self.slot_offset(slot, is_err);
        self.file.seek(SeekFrom::Start(offset))?;
        let byte_len = std::mem::size_of_val(data);
        // SAFETY: `f64` is plain old data with no padding; reinterpreting `&[f64]` as `&[u8]` is sound.
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len) };
        self.file.write_all(bytes)
    }

    fn read_vec(&mut self, slot: usize, is_err: bool, buf: &mut [f64]) -> std::io::Result<()> {
        let offset = self.slot_offset(slot, is_err);
        self.file.seek(SeekFrom::Start(offset))?;
        let byte_len = std::mem::size_of_val(buf);
        // SAFETY: `f64` is plain old data; writing exact bytes into initialized `&mut [f64]` is sound.
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, byte_len) };
        self.file.read_exact(bytes)
    }

    fn push_and_extrapolate(&mut self, fock: &Matrix2D, err: &Matrix2D) -> std::io::Result<Matrix2D> {
        let n_elems = self.n_elems();
        let slot = if self.slots.len() >= self.max_vecs {
            let evicted = self.slots.remove(0);
            self.dots.remove(0);
            for row in &mut self.dots {
                row.remove(0);
            }
            evicted
        } else {
            self.slots.len()
        };

        let m_prev = self.slots.len();
        let mut new_dots = Vec::with_capacity(m_prev + 1);
        let mut scratch = vec![0.0f64; n_elems];
        for i in 0..m_prev {
            let s_i = self.slots[i];
            self.read_vec(s_i, true, &mut scratch)?;
            let dot: f64 = scratch
                .iter()
                .zip(err.data.iter())
                .map(|(a, b)| a * b)
                .sum();
            new_dots.push(dot);
            self.dots[i].push(dot);
        }
        let self_dot: f64 = err.data.iter().map(|x| x * x).sum();
        new_dots.push(self_dot);
        self.dots.push(new_dots);
        self.slots.push(slot);

        self.write_vec(slot, false, &fock.data)?;
        self.write_vec(slot, true, &err.data)?;

        let total = self.slots.len();
        if total < 2 {
            return Ok(fock.clone());
        }

        for start in 0..(total - 1) {
            if let Some(coeffs) = DiisState::solve_pulay_coeffs(&self.dots, start, total) {
                let mut out = Matrix2D::zeros(self.rows, self.cols);
                for (k, &ck) in coeffs.iter().take(total - start).enumerate() {
                    let s_k = self.slots[start + k];
                    self.read_vec(s_k, false, &mut scratch)?;
                    for idx in 0..n_elems {
                        out.data[idx] += ck * scratch[idx];
                    }
                }
                return Ok(out);
            }
        }

        Ok(fock.clone())
    }
}

#[derive(Debug, Clone)]
pub struct DiisState {
    pub max_vecs: usize,
    pub fock_list: Vec<Matrix2D>,
    pub err_list: Vec<Matrix2D>,
    pub force_disk: bool,
    pub custom_dir: Option<PathBuf>,
    disk_store: Option<Arc<Mutex<DiskDiisStore>>>,
}

impl DiisState {
    pub fn new(max_vecs: usize) -> Self {
        let force_disk = std::env::var("CALCHEM_DIIS_DISK")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        Self {
            max_vecs: max_vecs.max(2),
            fock_list: Vec::new(),
            err_list: Vec::new(),
            force_disk,
            custom_dir: None,
            disk_store: None,
        }
    }

    pub fn from_scf_options(opts: &calchem_model::ScfOptions) -> Self {
        let mut s = Self::new(opts.diis_max_vecs);
        if opts.diis_disk {
            s.force_disk = true;
        }
        if let Some(ref d) = opts.diis_dir {
            s.custom_dir = Some(PathBuf::from(d));
        }
        s
    }

    /// Creates a disk-backed [`DiisState`] that spills Fock/error history vectors to a temporary
    /// ring-buffer file on disk and retains zero full history lists in RAM.
    pub fn new_disk(max_vecs: usize) -> Self {
        Self {
            max_vecs: max_vecs.max(2),
            fock_list: Vec::new(),
            err_list: Vec::new(),
            force_disk: true,
            custom_dir: None,
            disk_store: None,
        }
    }

    /// Creates a disk-backed [`DiisState`] in an explicit spill directory.
    pub fn new_disk_in_dir(
        dir: &std::path::Path,
        max_vecs: usize,
        rows: usize,
        cols: usize,
    ) -> std::io::Result<Self> {
        let store = DiskDiisStore::create_in_dir(dir, max_vecs, rows, cols)?;
        Ok(Self {
            max_vecs: max_vecs.max(2),
            fock_list: Vec::new(),
            err_list: Vec::new(),
            force_disk: true,
            custom_dir: Some(dir.to_path_buf()),
            disk_store: Some(Arc::new(Mutex::new(store))),
        })
    }

    pub fn is_disk_backed(&self) -> bool {
        self.disk_store.is_some()
    }

    pub fn disk_file_path(&self) -> Option<PathBuf> {
        self.disk_store
            .as_ref()
            .and_then(|s| s.lock().ok().map(|g| g.path.clone()))
    }

    fn should_use_disk(&self, n_elems: usize) -> bool {
        if self.force_disk || self.disk_store.is_some() {
            return true;
        }
        let max_ram_bytes: usize = std::env::var("CALCHEM_MAX_DIIS_RAM_BYTES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64 * 1024 * 1024); // 64 MB default threshold
        let est_bytes = 2usize
            .saturating_mul(self.max_vecs)
            .saturating_mul(n_elems)
            .saturating_mul(std::mem::size_of::<f64>());
        est_bytes > max_ram_bytes
    }

    /// Attempts a disk-backed or in-memory DIIS step, returning an explicit I/O error if
    /// disk-spilling was required (`should_use_disk`) and the underlying disk store failed.
    /// Never falls back to allocating in-RAM history when disk mode is active.
    pub fn try_push_and_extrapolate(
        &mut self,
        fock: &Matrix2D,
        err: &Matrix2D,
    ) -> std::io::Result<Matrix2D> {
        if self.should_use_disk(fock.data.len()) {
            self.fock_list.clear();
            self.err_list.clear();
            if self.disk_store.is_none() {
                let store = if let Some(ref dir) = self.custom_dir {
                    DiskDiisStore::create_in_dir(dir, self.max_vecs, fock.rows, fock.cols)?
                } else {
                    DiskDiisStore::create(self.max_vecs, fock.rows, fock.cols)?
                };
                self.disk_store = Some(Arc::new(Mutex::new(store)));
            }
            let arc_store = self.disk_store.as_ref().expect("disk store initialized");
            let mut guard = arc_store
                .lock()
                .map_err(|_| std::io::Error::other("DIIS disk store mutex poisoned"))?;
            return guard.push_and_extrapolate(fock, err);
        }
        Ok(self.push_and_extrapolate_ram(fock, err))
    }

    pub fn push_and_extrapolate(&mut self, fock: &Matrix2D, err: &Matrix2D) -> Matrix2D {
        if self.should_use_disk(fock.data.len()) {
            // Strictly forbid silent fallback to RAM when disk-spilling is active:
            // if disk I/O fails, keep RAM history empty and return the unextrapolated Fock matrix.
            return self
                .try_push_and_extrapolate(fock, err)
                .unwrap_or_else(|_| fock.clone());
        }
        self.push_and_extrapolate_ram(fock, err)
    }

    fn push_and_extrapolate_ram(&mut self, fock: &Matrix2D, err: &Matrix2D) -> Matrix2D {

        // Evict before cloning two full matrices, so a push never exceeds the history cap.
        while self.fock_list.len() >= self.max_vecs {
            self.fock_list.remove(0);
            self.err_list.remove(0);
        }
        self.fock_list.push(fock.clone());
        self.err_list.push(err.clone());
        while self.fock_list.len() > self.max_vecs {
            self.fock_list.remove(0);
            self.err_list.remove(0);
        }

        if self.fock_list.len() < 2 {
            return fock.clone();
        }

        // Try solving the Pulay B-matrix system; if near-singular, drop oldest vector(s)
        // and retry before falling back to the unextrapolated Fock matrix.
        let total = self.fock_list.len();
        for start in 0..(total - 1) {
            let f_slice = &self.fock_list[start..];
            let e_slice = &self.err_list[start..];
            if let Some(extrap) = Self::try_extrapolate_slice(f_slice, e_slice) {
                return extrap;
            }
        }

        fock.clone()
    }

    fn solve_pulay_coeffs(dots: &[Vec<f64>], start: usize, end: usize) -> Option<Vec<f64>> {
        let m = end.saturating_sub(start);
        if m < 2 {
            return None;
        }
        let dim = m + 1;
        let mut b = Matrix2D::zeros(dim, dim);
        let mut rhs = vec![0.0; dim];

        let mut max_diag: f64 = 0.0;
        for i in 0..m {
            b.set(i, m, -1.0);
            b.set(m, i, -1.0);
            for j in i..m {
                let dot = dots[start + i][start + j];
                if i == j && dot > max_diag {
                    max_diag = dot;
                }
                b.set(i, j, dot);
                b.set(j, i, dot);
            }
        }
        if !max_diag.is_finite() || max_diag <= 1e-30 {
            return None;
        }
        for i in 0..m {
            for j in 0..m {
                b.set(i, j, b.get(i, j) / max_diag);
            }
        }
        b.set(m, m, 0.0);
        rhs[m] = -1.0;

        let mut scale = 0.0_f64;
        for i in 0..m {
            for j in 0..m {
                scale = scale.max(b.get(i, j).abs());
            }
        }
        if !scale.is_finite() || scale <= f64::MIN_POSITIVE {
            return None;
        }
        for i in 0..m {
            for j in 0..m {
                b.set(i, j, b.get(i, j) / scale);
            }
        }

        let coeffs = solve_linear_system(&b, &rhs).ok()?;
        let weight_sum: f64 = coeffs.iter().take(m).sum();
        if !weight_sum.is_finite() || (weight_sum - 1.0).abs() > 1e-6 {
            return None;
        }
        if coeffs
            .iter()
            .take(m)
            .any(|&c| !c.is_finite() || c.abs() > 50.0)
        {
            return None;
        }
        Some(coeffs)
    }

    fn try_extrapolate_slice(focks: &[Matrix2D], errs: &[Matrix2D]) -> Option<Matrix2D> {
        let m = focks.len();
        if m < 2 {
            return None;
        }
        let mut dots = vec![vec![0.0; m]; m];
        for i in 0..m {
            for j in i..m {
                let dot: f64 = errs[i]
                    .data
                    .iter()
                    .zip(errs[j].data.iter())
                    .map(|(x, y)| x * y)
                    .sum();
                dots[i][j] = dot;
                dots[j][i] = dot;
            }
        }
        let coeffs = Self::solve_pulay_coeffs(&dots, 0, m)?;

        let mut out = Matrix2D::zeros(focks[0].rows, focks[0].cols);
        for (k, fk) in focks.iter().enumerate() {
            let ck = coeffs[k];
            for idx in 0..out.data.len() {
                out.data[idx] += ck * fk.data[idx];
            }
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disk_backed_diis_matches_in_memory_diis_with_zero_ram_lists() {
        let n = 12;
        let max_vecs = 4;
        let mut ram_diis = DiisState::new(max_vecs);
        let mut disk_diis = DiisState::new_disk(max_vecs);

        for step in 0..7 {
            let mut f = Matrix2D::zeros(n, n);
            let mut e = Matrix2D::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    let v_f = ((i * 3 + j * 5 + step * 7 + 1) as f64 * 0.23).cos();
                    let v_e = ((i * 11 + j * 2 + step * 13 + 3) as f64 * 0.41).sin()
                        * 0.1_f64.powi(step as i32);
                    f.set(i, j, v_f);
                    e.set(i, j, v_e);
                }
            }
            let out_ram = ram_diis.push_and_extrapolate(&f, &e);
            let out_disk = disk_diis.push_and_extrapolate(&f, &e);
            assert!(
                disk_diis.is_disk_backed()
                    && disk_diis.fock_list.is_empty()
                    && disk_diis.err_list.is_empty(),
                "Disk-backed DIIS must keep RAM fock_list and err_list empty"
            );
            for idx in 0..n * n {
                assert!(
                    (out_ram.data[idx] - out_disk.data[idx]).abs() < 1e-14,
                    "step {step} idx {idx}: ram={} vs disk={}",
                    out_ram.data[idx],
                    out_disk.data[idx]
                );
            }
        }
    }

    #[test]
    fn test_disk_diis_never_falls_back_to_ram_on_io_error() {
        // 1. Directory unavailable (file-as-directory or invalid path)
        let bad_dir = std::path::PathBuf::from("NUL:\\invalid\\calchem_diis_bad_dir");
        assert!(DiskDiisStore::create_in_dir(&bad_dir, 4, 4, 4).is_err());

        // 2. Inject real write/read failure into a live DiisState instance by replacing its file handle
        // with a read-only handle, and verify try_push_and_extrapolate returns Err without populating RAM
        let spill_dir = DiskDiisStore::resolve_spill_dir();
        let mut disk_diis = DiisState::new_disk_in_dir(&spill_dir, 4, 4, 4).unwrap();
        let backing_path = disk_diis.disk_file_path().expect("disk path must exist");
        assert!(backing_path.exists());

        {
            let arc = disk_diis.disk_store.as_ref().unwrap();
            let mut guard = arc.lock().unwrap();
            let ro_file = OpenOptions::new().read(true).write(false).open(&backing_path).unwrap();
            guard.file = ro_file;
        }

        let f = Matrix2D::identity(4);
        let e = Matrix2D::identity(4);
        let res = disk_diis.try_push_and_extrapolate(&f, &e);
        assert!(res.is_err(), "Read-only backing file must cause try_push_and_extrapolate to return Err");
        assert!(
            disk_diis.fock_list.is_empty() && disk_diis.err_list.is_empty(),
            "Disk-mode DIIS must never populate in-RAM history lists on I/O error"
        );

        // 3. Verify Drop cleans up the temporary file on disk
        drop(disk_diis);
        assert!(!backing_path.exists(), "DiskDiisStore Drop must remove spill file");
    }
}
