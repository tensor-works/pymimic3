Sure! Here is the revised table with the --mode and --timestep columns removed:

### Logistic Regression

| Task                  | --l2 | --C   | --no-grid-search |
|-----------------------|------|-------|------------------|
| In-hospital mortality |      | 0.001 |                  |
| Decompensation        |      |       | yes              |
| Length of Stay        |      |       | yes              |
| Phenotyping           |      |       | yes              |

### Standard LSTM

| Task                  | --dim | --depth | --batch_size | --dropout | --partition |
|-----------------------|-------|---------|--------------|-----------|-------------|
| In-hospital mortality | 16    | 2       | 8            | 0.3       |             |
| Decompensation        | 128   | 1       | 8            |           |             |
| Length of Stay        | 64    | 1       | 8            | 0.3       | custom      |
| Phenotyping           | 256   | 1       | 8            | 0.3       |             |

### Standard LSTM + Deep Supervision

| Task                  | --dim | --depth | --batch_size | --dropout | --target_repl_coef |
|-----------------------|-------|---------|--------------|-----------|--------------------|
| In-hospital mortality | 32    | 1       | 8            | 0.3       | 0.5                |
| Decompensation        | 128   | 1       | 8            | 0.3       |                    |
| Length of Stay        | 128   | 1       | 8            | 0.3       |                    |
| Phenotyping           | 256   | 1       | 8            | 0.3       | 0.5                |

### Channel-Wise LSTM

| Task                  | --dim | --depth | --batch_size | --dropout | --size_coef |
|-----------------------|-------|---------|--------------|-----------|-------------|
| In-hospital mortality | 8     | 1       | 8            | 0.3       | 4.0         |
| Decompensation        | 16    | 1       | 8            |           | 4.0         |
| Length of Stay        | 16    | 1       | 8            | 0.3       | 8.0         |
| Phenotyping           | 16    | 1       | 8            | 0.3       | 8.0         |

### Channel-Wise LSTM + Deep Supervision

| Task                  | --dim | --depth | --batch_size | --dropout | --size_coef | --target_repl_coef |
|-----------------------|-------|---------|--------------|-----------|-------------|--------------------|
| In-hospital mortality | 16    | 1       | 8            | 0.3       | 4.0         | 0.5                |
| Decompensation        | 16    | 1       | 8            |           | 8.0         |                    |
| Length of Stay        | 16    | 1       | 8            | 0.3       | 8.0         |                    |
| Phenotyping           | 16    | 1       | 8            | 0.3       | 8.0         | 0.5                |

### Multitask Standard LSTM

| Task                  | --dim | --depth | --batch_size | --dropout | --partition | --ihm_C | --decomp_C | --los_C | --pheno_C | --target_repl_coef |
|-----------------------|-------|---------|--------------|-----------|-------------|---------|------------|---------|-----------|--------------------|
| In-hospital mortality | 512   | 1       | 8            | 0.3       |             |         |            |         |           |                    |