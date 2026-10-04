# Distribution feeder data

Both files are copied verbatim from MATPOWER (`data/` directory) at commit
`2e0ef79a6c4526858ee923491d92781ad4507fb7`:

| File | Feeder | Original source | SHA-256 (first 16 hex) |
|---|---|---|---|
| `case33bw.m` | IEEE 33-bus | M. E. Baran and F. F. Wu, IEEE Trans. Power Del. 4(2):1401–1407, 1989, doi:10.1109/61.25627 | `b40831eeb444669a` |
| `case69.m` | IEEE 69-bus | M. E. Baran and F. F. Wu, IEEE Trans. Power Del. 4(1):725–734, 1989, doi:10.1109/61.19265 | `7bbdc8c39394eb6c` |

Branch impedances are in ohms and loads in kW/kvar (MATPOWER v2 convention);
`nflev.grid.network` converts them. MATPOWER notes that its case files are
derived from public sources and are not covered by its BSD code license.
