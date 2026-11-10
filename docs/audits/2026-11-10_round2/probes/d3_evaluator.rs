//! Standard two-body D3(BJ)/D3(0), in atomic units; ATM is not included.
//! CN interpolation uses the original pair reference tables, not atomic mixing rules.
//! Data attribution and corresponding source: ../data/d3/NOTICE.md.
use calchem_model::{Atom, CalcError};
use serde::{Deserialize, Serialize};
#[path = "d3_data.rs"]
mod data;
const A0_ANGSTROM: f64 = 0.529177210903;
const C6_BYTES: &[u8] = include_bytes!("../data/d3/c6.bin");

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DampingScheme {
    BeckeJohnson { s6: f64, s8: f64, a1: f64, a2: f64 },
    Zero { s6: f64, s8: f64, sr6: f64, sr8: f64, alpha6: f64, alpha8: f64 },
}
impl Default for DampingScheme {
    fn default() -> Self { Self::BeckeJohnson {s6:1.0,s8:1.9889,a1:0.3981,a2:4.4211} }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct D3Model { pub damping: DampingScheme, pub functional_name: String }

fn pair_index(a: usize,b: usize) -> usize { let hi=a.max(b); let lo=a.min(b); hi*(hi-1)/2+lo-1 }
pub fn covalent_radius_bohr(z:u8)->f64 { data::COVALENT_ANGSTROM[z as usize-1]*4.0/(3.0*A0_ANGSTROM) }
pub fn vdw_radius_bohr(z:u8)->f64 { data::VDW_PAIR_ANGSTROM[pair_index(z as usize,z as usize)]/(2.0*A0_ANGSTROM) }
fn r4r2(z:u8)->f64 { (0.5*(z as f64).sqrt()*data::R4_OVER_R2[z as usize-1]).sqrt() }
fn reference_c6(a:u8,b:u8,i:usize,j:usize)->f64 {
    let (r,c)=if a>b {(i,j)} else {(j,i)};
    let off=(pair_index(a as usize,b as usize)*49+c*7+r)*8;
    f64::from_le_bytes(C6_BYTES[off..off+8].try_into().expect("fixed D3 table layout"))
}
fn weights(z:u8,cn:f64)->([f64;7],[f64;7]) {
    let refs=&data::REFERENCE_CN[(z as usize-1)*7..z as usize*7];
    let mut w=[0.0;7]; let mut dw=[0.0;7];
    let mut nearest=f64::INFINITY;
    for &r in refs { if r>=0.0 { nearest=nearest.min((cn-r).powi(2)); } }
    let mut sum=0.0;
    for i in 0..7 { if refs[i]>=0.0 { w[i]=(-4.0*((cn-refs[i]).powi(2)-nearest)).exp();sum+=w[i]; } }
    let mut mean=0.0;
    for i in 0..7 { w[i]/=sum; dw[i]=-8.0*(cn-refs[i]); mean+=w[i]*dw[i]; }
    for i in 0..7 { dw[i]=w[i]*(dw[i]-mean); }
    (w,dw)
}
fn pair_c6(a:u8,b:u8,wa:&([f64;7],[f64;7]),wb:&([f64;7],[f64;7]))->(f64,f64,f64) {
    let(mut v,mut da,mut db)=(0.0,0.0,0.0);
    for i in 0..7 { for j in 0..7 { let c=reference_c6(a,b,i,j);v+=wa.0[i]*wb.0[j]*c;da+=wa.1[i]*wb.0[j]*c;db+=wa.0[i]*wb.1[j]*c; } }
    (v,da,db)
}
impl D3Model {
    // PARAMETER_CONSTRUCTORS

    pub fn energy(&self,atoms:&[Atom])->Result<f64,CalcError> { Ok(self.energy_and_gradient(atoms,false)?.0) }
    pub fn gradient(&self,atoms:&[Atom])->Result<Vec<[f64;3]>,CalcError> { Ok(self.energy_and_gradient(atoms,true)?.1) }
    pub fn energy_and_gradient(&self,atoms:&[Atom],compute_gradient:bool)->Result<(f64,Vec<[f64;3]>),CalcError> {
        for a in atoms { if !(1..=103).contains(&a.z) || !a.position_bohr.iter().all(|x|x.is_finite()) {
            return Err(CalcError::InvalidMolecule("D3 requires finite coordinates and elements Z=1..103".into()));
        } }
        let valid=match self.damping {
            DampingScheme::BeckeJohnson{s6,s8,a1,a2}=>[s6,s8,a1,a2].iter().all(|v|v.is_finite() && *v>=0.0) && a2>0.0,
            DampingScheme::Zero{s6,s8,sr6,sr8,alpha6,alpha8}=>[s6,s8,sr6,sr8,alpha6,alpha8].iter().all(|v|v.is_finite() && *v>=0.0) && sr6>0.0 && sr8>0.0 && alpha6>6.0 && alpha8>8.0,
        };
        if !valid { return Err(CalcError::InvalidMolecule("Invalid D3 damping parameters".into())); }
        let n=atoms.len(); let mut cn=vec![0.0;n];
        // O(N^2) geometry only; no AO tensor. 40 Bohr is the standard CN cutoff.
        for i in 0..n { for j in 0..i {
            let r=distance(atoms,i,j).1;
            if !r.is_finite() || r<1e-8 {return Err(CalcError::InvalidMolecule("D3 coincident atoms or non-finite distance".into()));}
            if r<=40.0 { let rc=covalent_radius_bohr(atoms[i].z)+covalent_radius_bohr(atoms[j].z);let f=1.0/(1.0+(-16.0*(rc/r-1.0)).exp());cn[i]+=f;cn[j]+=f; }
        } }
        let ws:Vec<_>=atoms.iter().zip(&cn).map(|(a,&c)|weights(a.z,c)).collect();
        let mut grad=vec![[0.0;3];n];let mut dedcn=vec![0.0;n];let mut energy=0.0;
        for i in 0..n { for j in 0..i {
            let (d,r)=distance(atoms,i,j);
            let (c6,dc_i,dc_j)=pair_c6(atoms[i].z,atoms[j].z,&ws[i],&ws[j]);
            let c8_ratio=3.0*r4r2(atoms[i].z)*r4r2(atoms[j].z);
            let mut kernel=0.0;let mut dkernel=0.0;
            match self.damping {
                DampingScheme::BeckeJohnson{s6,s8,a1,a2}=>{
                    let cutoff=a1*c8_ratio.sqrt()+a2;
                    for (power,scale) in [(6,s6),(8,s8*c8_ratio)] {
                        let denom=r.powi(power)+cutoff.powi(power);
                        kernel-=scale/denom;
                        dkernel+=scale*power as f64*r.powi(power-1)/(denom*denom);
                    }
                }
                DampingScheme::Zero{s6,s8,sr6,sr8,alpha6,alpha8}=>{
                    let r0=data::VDW_PAIR_ANGSTROM[pair_index(atoms[i].z as usize,atoms[j].z as usize)]/A0_ANGSTROM;
                    for (power,scale,sr,alpha) in [(6,s6,sr6,alpha6),(8,s8*c8_ratio,sr8,alpha8)] {
                        let damp=1.0/(1.0+6.0*(sr*r0/r).powf(alpha));
                        let factor=scale*damp/r.powi(power);
                        kernel-=factor;
                        dkernel+=factor/r*(power as f64-alpha*(1.0-damp));
                    }
                }
            }
            energy+=c6*kernel;
            if compute_gradient { dedcn[i]+=dc_i*kernel;dedcn[j]+=dc_j*kernel;
                for q in 0..3 {let f=c6*dkernel*d[q]/r;grad[i][q]+=f;grad[j][q]-=f;}
            }
        } }
        if compute_gradient { for i in 0..n { for j in 0..i {
            let(d,r)=distance(atoms,i,j);if r>40.0 {continue;}
            let rc=covalent_radius_bohr(atoms[i].z)+covalent_radius_bohr(atoms[j].z);
            let f=1.0/(1.0+(-16.0*(rc/r-1.0)).exp());
            let factor=-(dedcn[i]+dedcn[j])*16.0*rc/(r*r)*f*(1.0-f);
            for q in 0..3 {let v=factor*d[q]/r;grad[i][q]+=v;grad[j][q]-=v;}
        } } }
        if !energy.is_finite() || grad.iter().flatten().any(|x|!x.is_finite()) {
            return Err(CalcError::InvalidMolecule("D3 result exceeds floating-point range".into()));
        }
        Ok((energy,grad))
    }
}
fn distance(atoms:&[Atom],i:usize,j:usize)->([f64;3],f64) {
    let d=std::array::from_fn(|q|atoms[i].position_bohr[q]-atoms[j].position_bohr[q]);
    let r=d[0].hypot(d[1]).hypot(d[2]);(d,r)
}
