# CalChem 基准测试

用于 CalChem 性能基准测试的脚本和数据集。

## 目录结构

```
benchmarks/
├── sets/          # 基准测试分子集
│   ├── small/     # 小分子 (≤20 原子)
│   ├── medium/    # 中等分子 (20-100 原子)
│   └── large/     # 大分子 (>100 原子)
├── results/       # 基准测试结果 (JSONL)
└── run_bench.py   # 基准测试运行脚本
```

## 运行基准测试

```bash
python benchmarks/run_bench.py --set small --method b3lyp --basis def2-svp
```

## 与 Gaussian 对比

CalChem 以 D:\G16W\tests\gjf\ 中的 600+ 测试用例作为正确性回归基准。
