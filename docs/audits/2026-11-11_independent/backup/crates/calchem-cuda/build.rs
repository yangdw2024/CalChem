use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(no_cuda)");
    println!("cargo:rerun-if-changed=src/cuda_kernels.cu");

    let nvcc_check = Command::new("nvcc").arg("--version").output();
    let has_nvcc = match nvcc_check {
        Ok(out) => out.status.success(),
        Err(_) => false,
    };

    if !has_nvcc {
        println!("cargo:warning=NVCC not found; calchem-cuda will compile with CPU fallback stubs.");
        println!("cargo:rustc-cfg=no_cuda");
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cu_path = manifest_dir.join("src").join("cuda_kernels.cu");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let lib_name = if cfg!(target_os = "windows") {
        "calchem_cuda.lib"
    } else {
        "libcalchem_cuda.a"
    };
    let lib_path = out_dir.join(lib_name);
    let prebuilt_lib = manifest_dir.join("lib").join(lib_name);

    let mut compiled = false;

    // First attempt compilation via NVCC
    let ps_cmd = format!(
        "& nvcc -ccbin D:\\MICROS~3\\VC\\Tools\\MSVC\\14.51.36231\\bin\\Hostx64\\x64 --lib -O3 -Xcompiler /wd4819 '{}' -o '{}'",
        cu_path.display(),
        lib_path.display()
    );
    if let Ok(s) = Command::new("powershell").arg("-Command").arg(&ps_cmd).status() {
        if s.success() && lib_path.exists() {
            compiled = true;
        }
    }

    // Fall back to prebuilt lib if available
    if !compiled && prebuilt_lib.exists() {
        if std::fs::copy(&prebuilt_lib, &lib_path).is_ok() {
            compiled = true;
        }
    }

    if compiled {
        println!("cargo:rustc-link-search=native={}", out_dir.display());
        println!("cargo:rustc-link-lib=static=calchem_cuda");

        if let Ok(cuda_path) = env::var("CUDA_PATH") {
            let p = PathBuf::from(cuda_path).join("lib").join("x64");
            println!("cargo:rustc-link-search=native={}", p.display());
        } else {
            println!("cargo:rustc-link-search=native=D:\\NVIDIA_CUDA_Toolkit\\lib\\x64");
            println!("cargo:rustc-link-search=native=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.3\\lib\\x64");
        }
        println!("cargo:rustc-link-lib=static=cudart_static");
    } else {
        println!("cargo:warning=CUDA kernels could not be built; using CPU fallback");
        println!("cargo:rustc-cfg=no_cuda");
    }
}
