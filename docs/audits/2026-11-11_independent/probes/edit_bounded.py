from pathlib import Path
import difflib

def emit(p, s):
    delta = list(difflib.unified_diff(p.read_text(encoding='utf-8').splitlines(True), s.splitlines(True), n=3))
    print('*** Update File: ' + p.as_posix())
    for line in delta[2:]:
        if line.startswith('@@'):
            print('@@')
        else:
            print(line, end='')

print('*** Begin Patch')

root = Path(__file__).resolve().parents[4]
p = root / 'crates/calchem-dft/src/vxc.rs'
s = p.read_text(encoding='utf-8')
# At most K result matrices, with a serial reduction that reuses the first one.
# The old per-grid-chunk map/reduce allocated full matrices at every task boundary.
s = s.replace('const DENSITY_SCREEN: f64 = 1e-18;', '''const DENSITY_SCREEN: f64 = 1e-18;

fn bounded_grid_chunk_size(points: usize, n: usize, channels: usize) -> usize {
    let bytes = n.saturating_mul(n).saturating_mul(8).saturating_mul(channels).max(1);
    let budget = std::env::var("CALCHEM_MAX_DIRECT_BYTES").ok()
        .and_then(|v| v.parse::<usize>().ok()).unwrap_or(4 * 1024 * 1024 * 1024);
    let workers = rayon::current_num_threads().min((budget / bytes).max(1));
    points.div_ceil(workers).max(GRID_CHUNK_SIZE)
}''')
s = s.replace('.par_chunks(GRID_CHUNK_SIZE)', '.par_chunks(bounded_grid_chunk_size(grid.points.len(), n, 1))', 1)
s = s.replace('''        .reduce(
            || (0.0f64, vec![0.0f64; n * n]),''', '''        .collect::<Vec<_>>()
        .into_iter()
        .reduce(''', 1)
s = s.replace('''        );

    let mut v_mat''', '''        ).expect("nonempty grid");

    let mut v_mat''', 1)
s = s.replace('.par_chunks(GRID_CHUNK_SIZE)', '.par_chunks(bounded_grid_chunk_size(grid.points.len(), n, 2))', 1)
s = s.replace('''        .reduce(
            || (0.0f64, vec![0.0f64; n * n], vec![0.0f64; n * n]),''', '''        .collect::<Vec<_>>()
        .into_iter()
        .reduce(''', 1)
s = s.replace('''        );

    let mut va =''', '''        ).expect("nonempty grid");

    let mut va =''', 1)
emit(p, s)

p = root / 'crates/calchem-int/src/density_fitting.rs'
s = p.read_text(encoding='utf-8')
start = s.index('            let b_rows: Vec<Vec<f64>>')
end = s.index('            Some(b_flat)', start)
block = s[start:end]
block = block.replace('''            let b_rows: Vec<Vec<f64>> = aux_basis
                .shells
                .par_iter()
                .enumerate()
                .map(|(p_idx, p_shell)| {''', '''            let mut b_flat = vec![0.0; n_aux * n_ao_sq];
            let mut remainder = b_flat.as_mut_slice();
            let mut slices = Vec::with_capacity(aux_basis.shells.len());
            for shell in &aux_basis.shells {
                let (rows, rest) = remainder.split_at_mut(shell.n_functions() * n_ao_sq);
                slices.push(rows);
                remainder = rest;
            }
            aux_basis.shells.par_iter().zip(slices.into_par_iter()).enumerate()
                .for_each(|(p_idx, (p_shell, shell_rows))| {''')
block = block.replace('                    let mut shell_rows = vec![0.0f64; np * n_ao_sq];\n', '')
block = block[:block.index('                    shell_rows\n')] + '                });\n\n'
s = s[:start] + block + s[end:]
s = s.replace('let req_bytes = n_aux * n_ao_sq * 8;', '''let req_bytes = n_aux.checked_mul(n_ao_sq).and_then(|v| v.checked_mul(8))
            .ok_or_else(|| CalcError::UnsupportedBasis("RI-J tensor dimensions overflow".into()))?;''')
emit(p, s)

p = root / 'crates/calchem-int/src/direct_fock.rs'
s = p.read_text(encoding='utf-8')
# Reuse one exchange accumulator for output, as already done in the closed/open Fock paths.
s = s.replace('''        let mut final_k = vec![0.0f64; n * n];
        for w_buf in worker_buffers {''', '''        let mut final_k = worker_buffers.remove(0);
        for w_buf in worker_buffers {''')
emit(p, s)
print('*** End Patch')
