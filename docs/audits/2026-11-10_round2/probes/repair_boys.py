from pathlib import Path
R=Path(r'E:\YDW\Trae_Gaussian');p=R/'crates/calchem-int/src/boys.rs';s=p.read_text()
a=s.index('/// Numerical strategy:');b=s.index('pub fn boys_batch',a)
s=s[:a]+'''/// Positive Kummer series plus downward recurrence where upward recurrence
/// would subtract comparable numbers. At large T, upward recursion is used
/// only through n <= T; higher orders are evaluated by a positive series.
/// Batch evaluation allocates no auxiliary vector.
'''+s[b:]
a=s.index('    // For extremely large t');b=s.index('\n}\n\n/// Evaluates the single',a)
s=s[:a]+'''    if t.is_infinite() {
        out[..=max_n].fill(0.0);
        return;
    }
    let et = (-t).exp();
    if t > max_n as f64 + 40.0 || t >= 700.0 {
        out[0] = 0.5 * (PI / t).sqrt() * erf_approx(t.sqrt());
        let last_up = max_n.min(t as usize);
        for n in 1..=last_up {
            out[n] = ((2*n-1) as f64 * out[n-1] - et) * (0.5 / t);
        }
        // For n>T the Gaussian tail is not negligible, even if exp(-T) is
        // tiny. The old asymptotic-only branch produced unbounded relative error.
        for n in last_up+1..=max_n {
            out[n] = et * kummer_scaled(n,t);
        }
        return;
    }
    out[max_n] = et * kummer_scaled(max_n,t);
    for n in (1..=max_n).rev() {
        out[n-1] = (2.0*t*out[n] + et)/(2*n-1) as f64;
    }
}

/// exp(T) F_n(T), evaluated only where T <= n+40 and T < 700,
/// or n >= T. Kahan summation avoids loss in the positive series tail.
fn kummer_scaled(n:usize,t:f64)->f64 {
    let mut term=1.0/(2.0*n as f64+1.0);
    let mut sum=term; let mut correction=0.0;
    for k in 1..10000 {
        term *= t/(n as f64+k as f64+0.5);
        let y=term-correction;
        let next=sum+y;
        correction=(next-sum)-y;
        sum=next;
        if term<=sum*5e-17 {break;}
    }
    sum'''+s[b:]
out=R/'build_target/runtime/round2_20261110/repair_stage/crates/calchem-int/src/boys.rs';out.write_text(s)
