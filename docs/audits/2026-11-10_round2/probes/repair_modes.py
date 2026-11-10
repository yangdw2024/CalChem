from pathlib import Path
R=Path(r'E:\YDW\Trae_Gaussian');p=R/'crates/calchem-vib/src/modes.rs';s=p.read_text()
marker='    let masses_amu: Vec<f64> = mol.atoms.iter().map(|a| atomic_mass_amu(a.z)).collect();'
s=s.replace(marker,'''    if apt.rows!=3 || apt.cols!=dim || cart_hessian.data.iter().chain(&apt.data).any(|x|!x.is_finite()) {
        return Err(CalcError::InvalidMolecule("Hessian/APT must be finite, with APT shaped (3,3N)".into()));
    }
'''+marker)
a=s.index('    let (evals, evecs) = eigh_sym(&h_proj');b=s.index('    // Sort vibrational modes',a)
s=s[:a]+'''    // Diagonalize only the orthogonal complement of rigid translations/rotations.
    // Sorting the 3N spectrum by |lambda| can delete genuine floppy modes.
    let mut seeds=tr_basis.clone();
    for axis in 0..dim {let mut v=vec![0.0;dim];v[axis]=1.0;seeds.push(v);}
    let complete=gram_schmidt_ortho(dim,seeds);
    if complete.len()!=dim {return Err(CalcError::LinearAlgebra("Incomplete vibrational projection basis".into()));}
    let mut v=Matrix2D::zeros(dim,n_vib);
    for (j,col) in complete[n_tr..].iter().enumerate() {
        for i in 0..dim {v.set(i,j,col[i]);}
    }
    let hv=matmul(&matmul(&transpose(&v),&h_mw),&v);
    let (evals,small)=if n_vib>0 {eigh_sym(&hv,150,1e-14)?} else {(Vec::new(),Matrix2D::zeros(0,0))};
    let evecs=matmul(&v,&small);
    let mut vib_modes:Vec<(f64,Vec<f64>)>=(0..n_vib)
        .map(|j|(evals[j],(0..dim).map(|i|evecs.get(i,j)).collect())).collect();
    let trans_rot_modes:Vec<f64>=tr_basis.iter().map(|q| {
        let mut value=0.0;
        for i in 0..dim {for j in 0..dim {value+=q[i]*h_proj.get(i,j)*q[j];}}
        value
    }).collect();

'''+s[b:]
(R/'build_target/runtime/round2_20261110/repair_stage/crates/calchem-vib/src/modes.rs').write_text(s)
