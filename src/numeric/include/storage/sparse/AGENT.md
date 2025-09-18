# storage/sparse/ AGENT.md

## Purpose
Implement sparse storage formats for numeric containers: CSR, CSC, COO, and hybrid variants. These are low-level storage types used by high-level containers in `core/` and algorithms in `linear_algebra/`.

## Responsibilities
- Provide compact, cache-friendly storage structs with iterators over nonzeros.
- Support construction from triplets and from `sparse/` builders.
- Offer conversion utilities between formats.

## Files (planned)
- `csr_storage.h`, `csc_storage.h`, `coo_storage.h`, `hybrid_storage.h`.

## Collaborations
- `core/` high-level sparse_matrix/sparse_vector wrappers.
- `sparse/` builders and pattern utilities generate triplets/patterns.
- `linear_algebra/` sparse ops operate on these formats.

## Example APIs
```cpp
CSRStorage<double> csr{rows, cols, nnz};
csr.row_ptr = {...};
csr.col_ind = {...};
csr.values  = {...};

// Iterate nonzeros in row i
for (auto k = csr.row_ptr[i]; k < csr.row_ptr[i+1]; ++k) {
  int j = csr.col_ind[k];
  double aij = csr.values[k];
}

// Convert COO -> CSR
CSRStorage<double> to_csr(const COOStorage<double>& coo);
```

## Notes
- Keep headers minimal; format-specific algorithms remain in higher layers.

