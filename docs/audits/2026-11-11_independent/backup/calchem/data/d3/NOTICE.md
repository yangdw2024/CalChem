# Standard D3 reference data

Retrieved 2026-11-10 from https://github.com/dftd3/simple-dftd3 (main).
reference.f90, data_r4r2.f90, data_vdwrad.f90 and their converted tables are
LGPL-3.0-or-later (s-dftd3 authors); COPYING and COPYING.LESSER are included.
covrad.f90 is from https://github.com/grimme-lab/mctc-lib, Apache-2.0.
Exact downloaded revisions are identified by source_sha256.json; GitHub's
commit API was rate-limited, so no unverified commit id is asserted.

c6.bin is a lossless little-endian f64 conversion of reference_c6 in Fortran
column-major order (7,7,5356). d3_data.rs holds reference CN and atomic radii.
Original corresponding source tables are included here to make conversion
and redistribution auditable. No upstream numerical engine is called.
The Rust evaluator is a CalChem implementation of two-body D3; ATM is absent.
