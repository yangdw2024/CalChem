#[path = "../../../../crates/calchem-int/src/boys.rs"]
mod boys;
fn main(){
    println!("kind,n,x,value");
    for t in [0.0,1e-16,1e-14,1e-6,0.1,1.0,20.0,34.999999,35.0,35.000001,36.0,100.0,699.9,700.0,710.0,1000.0,1e6] {
        for n in (0..=128).chain([512,700,1024]) {
            println!("boys,{n},{t:.17e},{:.17e}",boys::boys(n,t));
        }
    }
}
