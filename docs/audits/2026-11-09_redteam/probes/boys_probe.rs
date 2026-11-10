#[path = "../../../../crates/calchem-int/src/boys.rs"]
mod boys;

fn main() {
    let ts: &[f64] = &[0.0, 1e-16, 9.999e-15, 1e-14, 1e-10, 1e-6, 0.01, 0.2, 1.0, 5.0, 20.0, 34.999999, 35.0, 35.000001, 36.0, 50.0, 100.0, 1000.0, 1e6];
    println!("kind,n,x,value");
    for &t in ts {
        for n in (0..=16).chain([24, 32, 64, 128]) {
            println!("boys,{n},{t:.17e},{:.17e}", boys::boys(n,t));
        }
    }
    for x in [0.0, 1e-12, 0.1, 0.46875, 0.5, 1.0, 2.0, 4.0, 5.0, 15.0, 26.6] {
        println!("erf,0,{x:.17e},{:.17e}", boys::erf_approx(x));
    }
}
