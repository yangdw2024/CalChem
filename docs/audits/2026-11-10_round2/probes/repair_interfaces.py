from pathlib import Path
import shutil,re
LIVE=Path(r'E:\YDW\Trae_Gaussian'); BASE=LIVE/'build_target/runtime/round2_20261110'; STAGE=BASE/'repair_stage'
changed=[]
def modify(name,func):
    s=(LIVE/name).read_text(encoding='utf-8-sig'); out=func(s); assert out!=s,name
    (STAGE/name).write_text(out,encoding='utf-8');changed.append(name)

def opt(s):
    a=s.index('    // 2. Solvent correction (e.g. C-PCM)'); b=s.index('\n    Ok((e, g))',a)
    s=s[:a]+s[b:]
    old='    let (mut e, mut g) = scf_energy_gradient(mol, bs, opts.reference, opts.grid_level, &opts.scf_options)?;'
    s=s.replace(old,'''    // Gas-phase derivatives are analytic. The optional SCRF surface uses explicit
    // central differences of *fully reconverged total energies*, including CDS.
    // A frozen gas density is not the derivative of either relaxed surface.
    let (mut e, mut g) = if opts.solvent.is_some() {
        let e = solvent_total_energy(mol, bs, opts)?;
        let h = 1e-4;
        let mut g = vec![[0.0; 3]; mol.atoms.len()];
        for a in 0..mol.atoms.len() { for q in 0..3 {
            let mut plus=mol.clone(); let mut minus=mol.clone();
            plus.atoms[a].position_bohr[q]+=h;minus.atoms[a].position_bohr[q]-=h;
            let bp=BasisSet::from_name(&bs.name,&plus)?;let bm=BasisSet::from_name(&bs.name,&minus)?;
            g[a][q]=(solvent_total_energy(&plus,&bp,opts)?-solvent_total_energy(&minus,&bm,opts)?)/(2.0*h);
        } }
        (e,g)
    } else { scf_energy_gradient(mol, bs, opts.reference, opts.grid_level, &opts.scf_options)? };''')
    a=s.index('        let func_str = match opts.reference');b=s.index('        e += e_d3;',a)
    s=s[:a]+'''        let d3 = dispersion_model(opts)?;
        let (e_d3, g_d3) = d3.energy_and_gradient(&mol.atoms, true)?;
'''+s[b:]
    a=s.index('    if opts.dispersion.is_none() && opts.solvent.is_none()');b=s.index('\n}\n',a)
    s=s[:a]+'''    let mut e=if opts.solvent.is_some() {solvent_total_energy(mol,bs,opts)?}
        else {scf_energy(mol,bs,opts.reference,opts.grid_level,&opts.scf_options)?};
    if opts.dispersion.is_some() { e+=dispersion_model(opts)?.energy(&mol.atoms)?; }
    Ok(e)'''+s[b:]
    s=s.replace('if let Some(ref disp_str) = opts.dispersion {','if opts.dispersion.is_some() {')
    idx=s.index('\n#[derive(Debug, Clone, Serialize, Deserialize)]\npub struct OptStep')
    s=s[:idx]+'''
fn dispersion_model(opts:&OptOptions)->Result<calchem_dft::D3Model,CalcError> {
    let functional=match opts.reference {
        Reference::Rks(xc)|Reference::Uks(xc)|Reference::DftAuto(xc)=>xc.canonical_name(),
        _=>"hf",
    };
    match opts.dispersion.as_deref().unwrap_or("").trim().to_ascii_lowercase().as_str() {
        "bj"|"d3bj"|"d3(bj)"=>calchem_dft::D3Model::d3bj(functional),
        "zero"|"d3zero"|"d3(0)"=>calchem_dft::D3Model::d3zero(functional),
        _=>Err(CalcError::UnsupportedMethod("Unknown D3 damping scheme".into())),
    }
}

fn solvent_total_energy(mol:&Molecule,bs:&BasisSet,opts:&OptOptions)->Result<f64,CalcError> {
    let solvent=calchem_solv::Solvent::from_name(opts.solvent.as_deref().unwrap_or(""))
        .ok_or_else(||CalcError::InvalidMolecule("Unknown solvent".into()))?;
    let (method,xc)=match opts.reference.resolve(mol) {
        Reference::Rhf|Reference::Auto=>("rhf",None),
        Reference::Rks(xc)=>("rks",Some(xc)),
        _=>return Err(CalcError::UnsupportedMethod("Open-shell SCRF gradients are not implemented; use an external backend".into())),
    };
    let solv_opts=calchem_solv::SolvationOptions {solvent,..Default::default()};
    let res=calchem_solv::run_scrf(mol,bs,&opts.scf_options,&solv_opts,method,xc)?;
    if !res.converged {return Err(CalcError::NotConverged {iterations:opts.scf_options.max_iter,delta_e:f64::INFINITY,rms_d:f64::INFINITY});}
    Ok(res.solvated_energy)
}
'''+s[idx:]
    return s
modify('crates/calchem-opt/src/lib.rs',opt)

def grad(s):
    a=s.index('    let opts = cavity_opts.cloned().unwrap_or_default();',s.index('pub fn compute_pcm_gradient_analytic'))
    b=s.index('\n}\n',a)
    s=s[:a]+'''    let _ = (mol, basis, density, solvent, cavity_opts);
    Err(CalcError::UnsupportedMethod(
        "Analytic PCM gradient is not implemented. Use compute_pcm_gradient_fd for a fixed-density partial derivative, or fully relaxed SCRF energy differences.".into()
    ))'''+s[b:]
    s=s.replace('/// Uses high-accuracy 2-point central finite differences on the self-consistent\n/// PCM solvation energy surface, guaranteeing absolute numerical fidelity across\n/// arbitrary cavity shapes and dielectric environments.','/// Central differences at FIXED AO density, including the CDS area term.\n/// This is a partial derivative; it excludes density response and is not the\n/// gradient of a relaxed gas-phase or SCRF energy. Cavity topology is piecewise smooth.')
    return s.replace('let e_cds = cds_kcal_mol / 627.509474;','let e_cds = cds_kcal_mol / crate::scrf::HARTREE_TO_KCAL_MOL;')
modify('crates/calchem-solv/src/gradient.rs',grad)

def native(s):
    s=s.replace('    "camb3lyp", "cam-b3lyp", "wb97xd", "wb97x-d", "wb97x",\n','')
    # Reject before broad backend fallbacks, both method and explicit XC/options.
    marker='    opts = options or {}\n'
    s=s.replace(marker,marker+'''    if any(_normalize_xc_token(str(v)) in {"camb3lyp", "wb97x", "wb97xd"}
           for v in (method, xc or "", opts.get("xc", ""))):
        raise UnsupportedMethodError(
            "Native CAM-B3LYP/wB97X(-D) is unavailable: range-separated ERIs are missing. "
            "Choose the Gaussian/ORCA backend explicitly."
        )
''',1)
    a=s.index('            # D3 色散能处理');b=s.index('\n            tot_energy =',a)
    s=s[:a]+'''            # Explicit standard two-body D3. Invalid corrections must not vanish silently.
            disp = opts.get("dispersion", opts.get("d3", None))
            d3_energy = None
            if disp:
                if not hasattr(rust_core, "run_d3_dispersion"):
                    raise UnsupportedMethodError("Native D3 extension is unavailable")
                d3_damp = "bj" if disp is True else str(disp).lower()
                d3_func = xc_canon or "hf"
                d3_calc = rust_core.run_d3_dispersion(
                    symbols, coords_bohr, damping=d3_damp, functional=d3_func
                )
                d3_energy = float(d3_calc.energy_hartree)
'''+s[b:]
    # Other errors must not launch a different method behind the user's back.
    a=s.index('        except Exception as e:',s.index('            # Explicit standard two-body D3.'))
    b=s.index('\n    def ',a)
    block=s[a:b]
    # Preserve next method boundary, replace generic catch only.
    s=s[:a]+'''        except Exception as e:
            raise UnsupportedMethodError(f"Native calculation failed without fallback: {e}") from e
'''+s[b:]
    return s
modify('calchem/backends/native.py',native)

print('\n'.join(changed))
