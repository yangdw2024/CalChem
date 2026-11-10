use calchem_int::symmetry::shell_quartets;
#[test]
fn lazy_quartets_match_canonical_definition() {
    for n in 0..7 {
        let mut expected=Vec::new();
        for i in 0..n {for j in i..n {for k in 0..n {for l in k..n {
            if i*n+j<=k*n+l {expected.push([i,j,k,l]);}
        }}}}
        assert_eq!(shell_quartets(n).collect::<Vec<_>>(),expected);
    }
    // This must not materialize the astronomically large full quartet space.
    assert_eq!(shell_quartets(1_000_000).take(2).collect::<Vec<_>>(),vec![[0,0,0,0],[0,0,0,1]]);
}
