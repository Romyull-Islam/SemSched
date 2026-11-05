# Measured results

Every table below is the output of the command above it, from one full rerun of REPRODUCE.md (2026-11-05). Throughput is decode tokens/s at B=128, a 512-token prompt and 256 decode tokens unless a table says otherwise; ratio is TierSched over the best baseline. Regenerate any of them with its command; nothing here is edited by hand.

## Table II: decode throughput and the bound

`python run_paper_tables.py`

```
==============================================================================
CPU-only (no accelerator)   Qwen2.5 72B, B=128, 256 decode steps   [decode t/s]
==============================================================================
Quant Memory        FlexGen       LIA    AimPod  LLMFlash      ITME TierSched    ratio    bound of bound
------------------------------------------------------------------------------
FP16  16H+32C          4.98      4.77      4.76      4.77      4.86      5.22   1.048x     5.26   99.2%
FP16  16H+48C          5.77      5.30      5.30      5.37      4.86      6.08   1.053x     6.15   98.8%
FP16  16H+64C          6.87      6.01      5.95      6.27      4.86      7.32   1.065x     7.41   98.7%
INT8  16H+32C         16.96     14.23     13.04     14.34     10.68     18.28   1.078x    18.65   98.0%
INT8  16H+48C         31.94     20.60     18.18     23.39     10.68     36.62   1.147x    38.49   95.1%
INT8  16H+64C         45.84     39.00     29.74     32.75     10.68     48.90   1.067x    48.99   99.8%
------------------------------------------------------------------------------
TierSched leads 6/6
==============================================================================
+RTX 5090 (28 GB)   Qwen2.5 72B, B=128, 256 decode steps   [decode t/s]
==============================================================================
Quant Memory        FlexGen       LIA    AimPod  LLMFlash      ITME TierSched    ratio    bound of bound
------------------------------------------------------------------------------
FP16  16H+32C          6.60      6.50      6.45      6.60      5.35      6.96   1.055x     7.05   98.7%
FP16  16H+48C          8.07      7.53      7.49      7.97      5.35      8.59   1.065x     8.76   98.1%
FP16  16H+64C         10.38      9.05      8.84     10.10      5.35     11.28   1.086x    11.56   97.5%
INT8  16H+32C         76.51     43.04     46.29     75.12     10.69     87.88   1.149x    88.13   99.7%
INT8  16H+48C         76.60     72.20     75.29     85.65     10.69     87.88   1.026x    88.13   99.7%
INT8  16H+64C         76.60     72.20     75.29     85.71     56.48     88.04   1.027x    88.13   99.9%
------------------------------------------------------------------------------
TierSched leads 6/6
```

## End-to-end breakdown (prefill, decode, wall clock)

`python run_paper_tables.py --detail`

```
==============================================================================================================
CPU-only (no accelerator)   Qwen2.5 72B, B=128, 512-token prompt, 256 decode steps
==============================================================================================================
QuantMemory   Policy     warmup s  staged prefill t/s prefill s decode t/s   step s  decode s  total s w-stall ms
--------------------------------------------------------------------------------------------------------------
FP16 16H+32C  FlexGen       0.000       -       0.799     641.1       4.98   25.703    6579.9   7221.0      1.331
FP16 16H+32C  LIA           0.000       -       0.798     641.5       4.77   26.859    6875.8   7517.3      1.331
FP16 16H+32C  AimPod        0.000       -       0.768     666.4       4.76   26.911    6889.3   7555.8      1.634
FP16 16H+32C  LLMFlash      0.000       -       0.799     640.8       4.77   26.829    6868.2   7509.0      1.503
FP16 16H+32C  ITME          0.000       -       0.798     641.5       4.86   26.354    6746.6   7388.1      0.000
FP16 16H+32C  TierSched     0.000       -       0.795     644.1       5.22   24.529    6279.3   6923.4      0.000
--------------------------------------------------------------------------------------------------------------
FP16 16H+48C  FlexGen       0.000       -       0.799     641.1       5.77   22.165    5674.3   6315.5      1.331
FP16 16H+48C  LIA           0.000       -       0.798     641.5       5.30   24.171    6187.9   6829.4      1.331
FP16 16H+48C  AimPod        0.000       -       0.772     663.5       5.30   24.140    6180.0   6843.5      1.634
FP16 16H+48C  LLMFlash      0.000       -       0.799     640.8       5.37   23.845    6104.3   6745.1      1.548
FP16 16H+48C  ITME          0.000       -       0.798     641.5       4.86   26.318    6737.5   7379.0      0.000
FP16 16H+48C  TierSched     0.000       -       0.795     643.8       6.08   21.055    5390.0   6033.8      0.000
--------------------------------------------------------------------------------------------------------------
FP16 16H+64C  FlexGen       0.000       -       0.799     641.1       6.87   18.628    4768.8   5409.9      1.331
FP16 16H+64C  LIA           0.000       -       0.798     641.5       6.01   21.315    5456.7   6098.3      1.331
FP16 16H+64C  AimPod        0.000       -       0.775     660.8       5.95   21.524    5510.2   6171.1      1.634
FP16 16H+64C  LLMFlash      0.000       -       0.799     640.8       6.27   20.411    5225.2   5866.0      1.575
FP16 16H+64C  ITME          0.000       -       0.798     641.5       4.86   26.318    6737.5   7379.0      0.000
FP16 16H+64C  TierSched     0.000       -       0.796     643.4       7.32   17.495    4478.7   5122.1      0.000
--------------------------------------------------------------------------------------------------------------
INT8 16H+32C  FlexGen       0.000       -       0.799     640.9      16.96    7.545    1931.6   2572.6      0.562
INT8 16H+32C  LIA           0.000       -       0.799     641.2      14.23    8.996    2303.1   2944.3      0.562
INT8 16H+32C  AimPod        0.000       -       0.789     649.3      13.04    9.816    2512.9   3162.2      0.817
INT8 16H+32C  LLMFlash      0.000       -       0.799     640.8      14.34    8.925    2284.7   2925.5      0.751
INT8 16H+32C  ITME          0.000       -       0.799     640.8      10.68   11.983    3067.6   3708.4      0.000
INT8 16H+32C  TierSched     0.000       -       0.798     641.9      18.28    7.001    1792.4   2434.3      0.000
--------------------------------------------------------------------------------------------------------------
INT8 16H+48C  FlexGen       0.000       -       0.799     640.9      31.94    4.008    1026.1   1667.0      0.562
INT8 16H+48C  LIA           0.000       -       0.799     641.2      20.60    6.213    1590.4   2231.6      0.562
INT8 16H+48C  AimPod        0.000       -       0.792     646.3      18.18    7.041    1802.4   2448.8      0.817
INT8 16H+48C  LLMFlash      0.000       -       0.799     640.8      23.39    5.474    1401.2   2042.0      0.774
INT8 16H+48C  ITME          0.000       -       0.799     640.8      10.68   11.983    3067.6   3708.4      0.000
INT8 16H+48C  TierSched     0.000       -       0.798     641.7      36.62    3.496     894.9   1536.5      0.000
--------------------------------------------------------------------------------------------------------------
INT8 16H+64C  FlexGen       0.000       -       0.799     641.0      45.84    2.793     714.9   1355.8      0.562
INT8 16H+64C  LIA           0.000       -       0.799     641.2      39.00    3.282     840.3   1481.5      0.562
INT8 16H+64C  AimPod        0.000       -       0.796     643.5      29.74    4.304    1101.8   1745.3      0.817
INT8 16H+64C  LLMFlash      0.000       -       0.799     640.8      32.75    3.908    1000.4   1641.2      0.787
INT8 16H+64C  ITME          0.000       -       0.799     640.8      10.68   11.983    3067.6   3708.4      0.000
INT8 16H+64C  TierSched     0.000       -       0.798     641.5      48.90    2.617     670.0   1311.6      0.000
--------------------------------------------------------------------------------------------------------------
==============================================================================================================
+RTX 5090 (28 GB)   Qwen2.5 72B, B=128, 512-token prompt, 256 decode steps
==============================================================================================================
QuantMemory   Policy     warmup s  staged prefill t/s prefill s decode t/s   step s  decode s  total s w-stall ms
--------------------------------------------------------------------------------------------------------------
FP16 16H+32C  FlexGen       0.000       -       8.140      62.9       6.60   19.403    4967.1   5030.0      1.331
FP16 16H+32C  LIA           0.000       -       8.075      63.4       6.50   19.682    5038.6   5102.0      1.331
FP16 16H+32C  AimPod        0.000       -       6.229      82.2       6.45   19.860    5084.1   5166.3      1.634
FP16 16H+32C  LLMFlash      0.000       -       8.169      62.7       6.60   19.394    4964.9   5027.6      0.958
FP16 16H+32C  ITME          0.000       -       8.169      62.7       5.35   23.941    6128.8   6191.5      0.000
FP16 16H+32C  TierSched     0.000       -       7.845      65.3       6.96   18.387    4707.1   4772.4      0.000
--------------------------------------------------------------------------------------------------------------
FP16 16H+48C  FlexGen       0.000       -       8.140      62.9       8.07   15.865    4061.5   4124.4      1.331
FP16 16H+48C  LIA           0.000       -       8.075      63.4       7.53   16.995    4350.7   4414.1      1.331
FP16 16H+48C  AimPod        0.000       -       6.459      79.3       7.49   17.089    4374.8   4454.1      1.634
FP16 16H+48C  LLMFlash      0.000       -       8.169      62.7       7.97   16.063    4112.2   4174.9      1.084
FP16 16H+48C  ITME          0.000       -       8.169      62.7       5.35   23.941    6128.8   6191.5      0.000
FP16 16H+48C  TierSched     0.000       -       7.887      64.9       8.59   14.895    3813.2   3878.1      0.000
--------------------------------------------------------------------------------------------------------------
FP16 16H+64C  FlexGen       0.000       -       8.140      62.9      10.38   12.328    3155.9   3218.8      1.331
FP16 16H+64C  LIA           0.000       -       8.075      63.4       9.05   14.139    3619.6   3683.0      1.331
FP16 16H+64C  AimPod        0.000       -       6.682      76.6       8.84   14.473    3705.1   3781.8      1.634
FP16 16H+64C  LLMFlash      0.000       -       8.169      62.7      10.10   12.671    3243.7   3306.4      1.173
FP16 16H+64C  ITME          0.000       -       8.169      62.7       5.35   23.941    6128.8   6191.5      0.000
FP16 16H+64C  TierSched     0.000       -       7.928      64.6      11.28   11.352    2906.1   2970.7      0.000
--------------------------------------------------------------------------------------------------------------
INT8 16H+32C  FlexGen       0.000       -       8.155      62.8      76.51    1.673     428.3    491.0      0.562
INT8 16H+32C  LIA           0.000       -       8.122      63.0      43.04    2.974     761.3    824.3      0.562
INT8 16H+32C  AimPod        0.000       -       7.875      65.0      46.29    2.765     707.8    772.8      0.817
INT8 16H+32C  LLMFlash      0.000       -       8.169      62.7      75.12    1.704     436.2    498.9      0.479
INT8 16H+32C  ITME          0.000       -       8.169      62.7      10.69   11.971    3064.6   3127.2      0.000
INT8 16H+32C  TierSched     0.000       -       8.109      63.1      87.88    1.457     372.9    436.0      0.000
--------------------------------------------------------------------------------------------------------------
INT8 16H+48C  FlexGen       0.000       -       8.166      62.7      76.60    1.671     427.8    490.5      0.562
INT8 16H+48C  LIA           0.000       -       8.160      62.7      72.20    1.773     453.9    516.6      0.562
INT8 16H+48C  AimPod        0.000       -       8.166      62.7      75.29    1.700     435.2    497.9      0.817
INT8 16H+48C  LLMFlash      0.000       -       8.169      62.7      85.65    1.494     382.6    445.2      0.502
INT8 16H+48C  ITME          0.000       -       8.169      62.7      10.69   11.971    3064.6   3127.2      0.000
INT8 16H+48C  TierSched     0.000       -       8.109      63.1      87.88    1.457     372.9    436.0      0.000
--------------------------------------------------------------------------------------------------------------
INT8 16H+64C  FlexGen       0.000       -       8.166      62.7      76.60    1.671     427.8    490.5      0.562
INT8 16H+64C  LIA           0.000       -       8.160      62.7      72.20    1.773     453.9    516.6      0.562
INT8 16H+64C  AimPod        0.000       -       8.166      62.7      75.29    1.700     435.2    497.9      0.817
INT8 16H+64C  LLMFlash      0.000       -       8.169      62.7      85.71    1.493     382.3    445.0      0.506
INT8 16H+64C  ITME          0.000       -       8.169      62.7      56.48    2.266     580.2    642.9      0.000
INT8 16H+64C  TierSched     0.000       -       8.109      63.1      88.04    1.454     372.2    435.3      0.000
--------------------------------------------------------------------------------------------------------------
```

## Prefill throughput

`python run_paper_tables.py --prefill`

```
==============================================================================
CPU-only (no accelerator)   Qwen2.5 72B, B=128, 256 decode steps   [prefill t/s]
==============================================================================
Quant Memory        FlexGen       LIA    AimPod  LLMFlash      ITME TierSched    ratio
------------------------------------------------------------------------------
FP16  16H+32C          0.80      0.80      0.77      0.80      0.80      0.79   0.995x
FP16  16H+48C          0.80      0.80      0.77      0.80      0.80      0.80   0.995x
FP16  16H+64C          0.80      0.80      0.77      0.80      0.80      0.80   0.996x
INT8  16H+32C          0.80      0.80      0.79      0.80      0.80      0.80   0.998x
INT8  16H+48C          0.80      0.80      0.79      0.80      0.80      0.80   0.999x
INT8  16H+64C          0.80      0.80      0.80      0.80      0.80      0.80   0.999x
------------------------------------------------------------------------------
TierSched leads 0/6
==============================================================================
+RTX 5090 (28 GB)   Qwen2.5 72B, B=128, 256 decode steps   [prefill t/s]
==============================================================================
Quant Memory        FlexGen       LIA    AimPod  LLMFlash      ITME TierSched    ratio
------------------------------------------------------------------------------
FP16  16H+32C          8.14      8.07      6.23      8.17      8.17      7.85   0.960x
FP16  16H+48C          8.14      8.07      6.46      8.17      8.17      7.89   0.965x
FP16  16H+64C          8.14      8.07      6.68      8.17      8.17      7.93   0.970x
INT8  16H+32C          8.15      8.12      7.88      8.17      8.17      8.11   0.993x
INT8  16H+48C          8.17      8.16      8.17      8.17      8.17      8.11   0.993x
INT8  16H+64C          8.17      8.16      8.17      8.17      8.17      8.11   0.993x
------------------------------------------------------------------------------
TierSched leads 0/6
```

## Decode length (Sec. V-A)

`python realistic_runs.py decode`

```
Decode length: decode ratio and end-to-end ratio against the best baseline on each metric
setup      plat quant  decode  decode ratio  end-to-end
paper      CPU  fp16       16         1.05x       1.01x
paper      CPU  fp16      128         1.05x       1.04x
paper      CPU  fp16      512         1.06x       1.05x
paper      CPU  fp16     2048         1.07x       1.07x
paper      CPU  int8       16         1.17x       1.01x
paper      CPU  int8      128         1.16x       1.06x
paper      CPU  int8      512         1.13x       1.10x
paper      CPU  int8     2048         1.17x       1.16x
paper      GPU  fp16       16         1.07x       1.04x
paper      GPU  fp16      128         1.07x       1.06x
paper      GPU  fp16      512         1.06x       1.06x
paper      GPU  fp16     2048         1.05x       1.05x
paper      GPU  int8       16         1.03x       1.00x
paper      GPU  int8      128         1.03x       1.02x
paper      GPU  int8      512         1.02x       1.02x
paper      GPU  int8     2048         1.19x       1.19x
realistic  CPU  fp16       16         1.07x       1.02x
realistic  CPU  fp16      128         1.08x       1.06x
realistic  CPU  fp16      512         1.08x       1.07x
realistic  CPU  fp16     2048         1.06x       1.06x
realistic  CPU  int8       16         1.39x       1.03x
realistic  CPU  int8      128         1.38x       1.16x
realistic  CPU  int8      512         1.35x       1.27x
realistic  CPU  int8     2048         1.29x       1.27x
realistic  GPU  fp16       16         1.01x       1.00x
realistic  GPU  fp16      128         1.02x       1.01x
realistic  GPU  fp16      512         1.02x       1.02x
realistic  GPU  fp16     2048         1.01x       1.01x
realistic  GPU  int8       16         1.08x       1.03x
realistic  GPU  int8      128         1.08x       1.06x
realistic  GPU  int8      512         1.08x       1.07x
realistic  GPU  int8     2048         1.07x       1.07x
```

## Equal-footing baselines

`python run_paper_tables.py --equal-footing`

```
====================================================================================================
Equal footing: every policy at its best setting  (CPU-only)
====================================================================================================
Quant Memory                  best static baseline        dynamic (Belady)         TierSched  vs static vs dynamic
----------------------------------------------------------------------------------------------------
FP16  16H+32C       5.01 (LLMFlash K=32 r=0.0%/3.0% b=8)     5.02 (K=4 KV=host b=8)    5.22 (K=32)      1.04x      1.04x
FP16  16H+48C       5.80 (FlexGen K=32 r=0.0%/0.0% b=8)     5.74 (K=4 KV=cxl b=8)    6.08 (K=32)      1.05x      1.06x
FP16  16H+64C       6.91 (FlexGen K=32 r=0.0%/0.0% b=8)     6.71 (K=8 KV=cxl b=8)    7.32 (K=32)      1.06x      1.09x
INT8  16H+32C      17.07 (FlexGen K=32 r=0.0%/0.0% b=8)    16.12 (K=16 KV=host b=8)   18.28 (K=32)      1.07x      1.13x
INT8  16H+48C      32.31 (FlexGen K=32 r=0.0%/0.0% b=8)    26.14 (K=16 KV=cxl b=8)   36.62 (K=32)      1.13x      1.40x
INT8  16H+64C      48.39 (LLMFlash K=32 r=0.0%/10.0% b=8)    48.51 (K=32 KV=cxl b=1)   48.90 (K=32)      1.01x      1.01x
====================================================================================================
Equal footing: every policy at its best setting  (+RTX 5090)
====================================================================================================
Quant Memory                  best static baseline        dynamic (Belady)         TierSched  vs static vs dynamic
----------------------------------------------------------------------------------------------------
FP16  16H+32C       6.61 (LLMFlash K=32 r=0.0%/1.5% b=8)     6.59 (K=4 KV=host b=8)    6.96 (K=32)      1.05x      1.06x
FP16  16H+48C       8.07 (FlexGen K=32 r=0.0%/0.0% b=8)     7.88 (K=8 KV=host b=8)    8.59 (K=32)      1.06x      1.09x
FP16  16H+64C      10.38 (FlexGen K=32 r=0.0%/0.0% b=8)     9.77 (K=8 KV=cxl b=8)   11.28 (K=32)      1.09x      1.15x
INT8  16H+32C      81.05 (LLMFlash K=32 r=0.0%/4.5% b=8)    76.53 (K=16 KV=cxl b=1)   87.88 (K=32)      1.08x      1.15x
INT8  16H+48C      85.77 (AimPod K=32 r=0.0%/21.3% b=8)    87.02 (K=4096 KV=cxl b=1)   87.88 (K=32)      1.02x      1.01x
INT8  16H+64C      85.71 (LLMFlash K=32 r=0.0%/37.5% b=1)    87.02 (K=4096 KV=cxl b=1)   88.04 (K=32)      1.03x      1.01x
```

## Where the lead over the tuned baselines comes from

`python attribution.py`

```
cell                   tuned  contiguous, one block,       + transfer blocks              + striping  + host or device first  + KV tier, device size
CPU fp16 16H+32C        5.01          5.08 (1.014x)          5.13 (1.024x)          5.19 (1.036x)          5.19 (1.036x)          5.22 (1.042x)
CPU fp16 16H+48C        5.80          5.85 (1.008x)          5.91 (1.019x)          6.06 (1.045x)          6.06 (1.045x)          6.08 (1.047x)
CPU fp16 16H+64C        6.91          6.91 (0.999x)          6.99 (1.011x)          7.27 (1.051x)          7.27 (1.051x)          7.32 (1.058x)
CPU int8 16H+32C       17.07         16.90 (0.990x)         17.21 (1.008x)         18.24 (1.069x)         18.24 (1.069x)         18.28 (1.071x)
CPU int8 16H+48C       32.31         30.02 (0.929x)         31.11 (0.963x)         36.62 (1.133x)         36.62 (1.133x)         36.62 (1.133x)
CPU int8 16H+64C       48.39         46.81 (0.967x)         47.46 (0.981x)         48.53 (1.003x)         48.90 (1.011x)         48.90 (1.011x)
GPU fp16 16H+32C        6.61          6.77 (1.024x)          6.85 (1.037x)          6.93 (1.048x)          6.93 (1.048x)          6.96 (1.053x)
GPU fp16 16H+48C        8.07          8.20 (1.017x)          8.33 (1.032x)          8.58 (1.063x)          8.58 (1.063x)          8.59 (1.065x)
GPU fp16 16H+64C       10.38         10.46 (1.007x)         10.63 (1.023x)         11.21 (1.079x)         11.21 (1.079x)         11.28 (1.086x)
GPU int8 16H+32C       81.05         67.65 (0.835x)         68.81 (0.849x)         87.23 (1.076x)         87.88 (1.084x)         87.88 (1.084x)
GPU int8 16H+48C       85.77         82.68 (0.964x)         83.38 (0.972x)         87.10 (1.015x)         87.80 (1.024x)         87.88 (1.025x)
GPU int8 16H+64C       85.71         83.43 (0.973x)         84.19 (0.982x)         87.10 (1.016x)         88.04 (1.027x)         88.04 (1.027x)
contiguous, one block, even split  vs tuned: median 0.994 min 0.835 max 1.024, leads 5/12
+ transfer blocks                  vs tuned: median 1.009 min 0.849 max 1.037, leads 7/12
+ striping                         vs tuned: median 1.050 min 1.003 max 1.133, leads 11/12
+ host or device first             vs tuned: median 1.050 min 1.011 max 1.133, leads 12/12
+ KV tier, device size, spreading (full) vs tuned: median 1.056 min 1.011 max 1.133, leads 12/12
```

## Table III: robustness

`python sensitivity.py all`

```
       drop 0%: speedup min 1.03 median 1.06 max 1.15, wins 12/12, failed 0
      drop 10%: speedup min 1.00 median 1.06 max 1.13, wins 11/12, failed 0
      drop 25%: speedup min 0.97 median 1.06 max 1.12, wins 11/12, failed 0
      drop 50%: speedup min 0.93 median 1.06 max 1.09, wins 10/12, failed 0
     drop 100%: speedup min 0.81 median 1.04 max 1.08, wins 8/12, failed 0
      2.5 GB/s: speedup min 1.02 median 1.06 max 1.26, wins 12/12, failed 0
        5 GB/s: speedup min 1.03 median 1.06 max 1.15, wins 12/12, failed 0
      7.5 GB/s: speedup min 1.03 median 1.06 max 1.12, wins 12/12, failed 0
        0.5 us: speedup min 1.03 median 1.07 max 1.15, wins 12/12, failed 0
      1.547 us: speedup min 1.03 median 1.06 max 1.15, wins 12/12, failed 0
          5 us: speedup min 1.02 median 1.06 max 1.15, wins 12/12, failed 0
         20 us: speedup min 1.02 median 1.06 max 1.14, wins 12/12, failed 0
         50 us: speedup min 1.02 median 1.05 max 1.25, wins 12/12, failed 0
        100 us: speedup min 1.02 median 1.04 max 1.19, wins 12/12, failed 0
   prompt 2048: speedup min 1.01 median 1.05 max 1.11, wins 12/12, failed 0
   prompt 8192: speedup min 1.01 median 1.05 max 1.20, wins 12/12, failed 0
  prompt 32768: speedup min 1.00 median 1.00 max 1.02, wins 2/12, failed 0
pass-through 1: speedup min 1.03 median 1.06 max 1.15, wins 12/12, failed 0
pass-through 0: speedup min 1.05 median 1.08 max 1.18, wins 12/12, failed 0
stream compute 0: speedup min 1.03 median 1.06 max 1.15, wins 12/12, failed 0
stream compute 1: speedup min 1.01 median 1.06 max 1.16, wins 12/12, failed 0
    gpu link 0: speedup min 1.03 median 1.06 max 1.15, wins 12/12, failed 0
 gpu link 63e9: speedup min 1.03 median 1.06 max 1.15, wins 12/12, failed 0
 gpu link 55e9: speedup min 1.03 median 1.06 max 1.15, wins 12/12, failed 0
served 448, planned 512: speedup min 1.02 median 1.05 max 1.12, wins 12/12, failed 0
served 576, planned 512: speedup min 1.01 median 1.06 max 1.18, wins 12/12, failed 0
served 512, planned 576: speedup min 1.03 median 1.05 max 1.12, wins 12/12, failed 0
served 448, planned 576: speedup min 1.02 median 1.04 max 1.09, wins 12/12, failed 0
served 512, planned 640: speedup min 1.02 median 1.04 max 1.11, wins 12/12, failed 0
pass-through x2: speedup min 1.02 median 1.06 max 1.15, wins 12/12, failed 0
pass-through x2, planned for x1: speedup min 0.88 median 1.06 max 1.15, wins 9/12, failed 0
one request at a time: speedup min 1.01 median 1.04 max 1.14, wins 12/12, failed 0
one request at a time, planned for concurrent: speedup min 0.58 median 1.01 max 1.09, wins 8/12, failed 0
NAND-only staging 0: speedup min 1.03 median 1.06 max 1.15, wins 12/12, failed 0
NAND-only staging 1: speedup min 1.03 median 1.06 max 1.16, wins 12/12, failed 0
in issue order: speedup min 1.02 median 1.07 max 1.15, wins 12/12, failed 0
in issue order, planned for the default: speedup min 0.83 median 1.05 max 1.09, wins 9/12, failed 0
host link 31.5 GB/s: speedup min 1.03 median 1.06 max 1.15, wins 12/12, failed 0
host link 27 GB/s: speedup min 1.03 median 1.06 max 1.15, wins 12/12, failed 0
host 38.4 GB/s: speedup min 1.03 median 1.06 max 1.15, wins 12/12, failed 0
 host 161 GB/s: speedup min 1.03 median 1.06 max 1.16, wins 12/12, failed 0
served 128, planned 512: speedup min 0.76 median 0.94 max 1.01, wins 2/12, failed 0
served 256, planned 512: speedup min 0.86 median 0.99 max 1.05, wins 5/12, failed 0
served 1024, planned 512: speedup min 0.60 median 1.05 max 1.15, wins 9/12, failed 0
served 2048, planned 512: speedup min 0.38 median 1.02 max 1.22, wins 9/12, failed 0
served 128, replanned: speedup min 1.01 median 1.06 max 1.12, wins 12/12, failed 0
served 256, replanned: speedup min 1.02 median 1.07 max 1.17, wins 12/12, failed 0
served 1024, replanned: speedup min 1.02 median 1.05 max 1.15, wins 12/12, failed 0
served 2048, replanned: speedup min 1.00 median 1.04 max 1.27, wins 10/12, failed 0
```

## Table IV: mechanism ablation

`python ablation_passthrough.py`

```
cell                  base            full  one block per      no striping  no striping, o  no striping or  reserve split   no smaller dev  search disable     filled, K=0  filled, K=0, o  contiguous, se  contiguous, ra
CPU fp16 32C          4.98            5.22            5.11            5.16            5.09            5.16            5.22            5.22            4.96            4.96            4.96            5.14            5.05
CPU fp16 48C          5.77            6.08            5.96            6.01            5.93            6.01            6.08            6.08            5.75            5.75            5.75            5.73            5.85
CPU fp16 64C          6.87            7.32            7.13            7.19            7.06            7.19            7.32            7.32            6.83            6.83            6.83            6.59            6.97
CPU int8 32C         16.96           18.28           17.72           17.86           17.51           17.86           18.28           18.28           15.56           15.56           15.56           15.86           16.95
CPU int8 48C         31.94           36.62           34.21           34.31           33.04           34.31           36.62           36.62           25.59           25.59           25.59           24.17           29.02
CPU int8 64C         45.84           48.90           48.91           48.78           48.58           48.78           48.53           48.90           32.93           32.93           32.94           44.46           47.79
GPU fp16 32C          6.60            6.96            6.96            6.91            6.78            6.91            6.96            6.96            6.92            6.92            6.92            6.53            6.68
GPU fp16 48C          8.07            8.59            8.59            8.50            8.34            8.50            8.59            8.59            8.56            8.56            8.56            7.71            8.08
GPU fp16 64C         10.38           11.28           11.24           11.15           10.84           11.15           11.28           11.28           11.21           11.21           11.21            9.34           10.18
GPU int8 32C         76.51           87.88           86.98           78.51           78.42           78.51           87.23           87.88           75.61           75.61           75.62           49.41           63.13
GPU int8 48C         85.65           87.88           87.83           87.51           87.59           87.29           87.23           87.80           80.56           80.56           80.58           81.89           84.49
GPU int8 64C         85.71           88.04           88.06           87.51           87.59           87.29           87.23           88.04           80.56           80.56           80.58           84.51           84.51
full                       vs full: median 1.000 min 1.000 max 1.000 | beats best baseline in 12/12
one block per sub-layer    vs full: median 0.993 min 0.934 max 1.000 | beats best baseline in 12/12
no striping                vs full: median 0.989 min 0.893 max 0.997 | beats best baseline in 12/12
no striping, one block     vs full: median 0.972 min 0.892 max 0.997 | beats best baseline in 12/12
no striping or spreading   vs full: median 0.989 min 0.893 max 0.997 | beats best baseline in 12/12
reserve split evenly only  vs full: median 1.000 min 0.991 max 1.000 | beats best baseline in 12/12
no smaller device          vs full: median 1.000 min 0.999 max 1.000 | beats best baseline in 12/12
search disabled (filled)   vs full: median 0.925 min 0.673 max 0.996 | beats best baseline in 3/12
filled, K=0                vs full: median 0.925 min 0.673 max 0.996 | beats best baseline in 3/12
filled, K=0, one block     vs full: median 0.925 min 0.674 max 0.996 | beats best baseline in 3/12
contiguous, semantic order vs full: median 0.905 min 0.562 max 0.985 | beats best baseline in 1/12
contiguous, random order   vs full: median 0.956 min 0.718 max 0.977 | beats best baseline in 6/12
```

## Per-configuration search against one fixed reserve

`python fixed_reserve.py`

```
Best single reserve over the twelve: 1.00% of host + device, split device; against the search: worst 0.984, median 0.996
Filled (no reserve) against the search: worst 0.673, median 0.927
sweep          cells   worst  median  wins search  wins pinned  wins minimum  losses search  losses pinned
model scale       16   0.776   0.998           13           10             7              2              4
batch             84   0.970   0.994           48           47            47             36             36
prompt            36   0.982   0.998           26           24            24              0              0
NAND rate         24   0.982   0.997           24           24            24              0              0
NAND latency      36   0.958   0.998           36           36            36              0              0
all              196   0.776   0.996          147          141           138             38             40
Largest losses of the pinned reserve against the search:
  GPU int8 Llama 13B               search    410.14  pinned    318.30 (0.776)  best baseline    379.99
  CPU int8 Mistral 7B              search    628.89  pinned    596.09 (0.948)  best baseline    594.58
  GPU int8 32C 100 us              search     58.72  pinned     56.23 (0.958)  best baseline     49.34
  CPU fp16 Llama 405B              search      0.79  pinned      0.77 (0.965)  best baseline      0.78
  CPU fp16 Mistral 7B              search    321.19  pinned    311.15 (0.969)  best baseline    304.48
  GPU fp16 64C B=1                 search      0.18  pinned      0.17 (0.970)  best baseline      0.54
```

## KV-cache placement

`python run_paper_tables.py --kv-tier`

```
==============================================================================
KV cache tier — TierSched, CPU-only, B=128
==============================================================================
Quant Memory        KV in cxl  KV in hostKV in stripe    best
------------------------------------------------------------------------------
FP16  16H+32C            5.19        5.19        5.22  stripe
FP16  16H+48C            6.06        6.06        6.08  stripe
FP16  16H+64C            7.27        7.27        7.32  stripe
INT8  16H+32C           18.24       18.22       18.28  stripe
INT8  16H+48C           36.62       36.54       36.47     cxl
INT8  16H+64C           48.90       47.09       48.80     cxl
==============================================================================
KV cache tier — TierSched, +RTX 5090 (28 GB), B=128
==============================================================================
Quant Memory        KV in cxl  KV in hostKV in stripe   KV in gpu    best
------------------------------------------------------------------------------
FP16  16H+32C            6.93        6.93        6.96        6.91  stripe
FP16  16H+48C            8.58        8.58        8.59        8.53  stripe
FP16  16H+64C           11.21       11.21       11.28       11.19  stripe
INT8  16H+32C           87.88       82.19       87.02       82.19     cxl
INT8  16H+48C           87.88       82.19       87.02       82.19     cxl
INT8  16H+64C           88.04       82.19       87.02       82.19     cxl
```

## Batch size (fig_batch.pdf)

`python run_paper_tables.py --batch-sweep`

```
==============================================================================
CPU-only   Qwen2.5 72B FP16  16H+32C   batch sweep
==============================================================================
   B     FlexGen       LIA    AimPod  LLMFlash      ITME TierSched    ratio   active_frac
------------------------------------------------------------------------------
   1        0.05      0.04      0.05      0.15      0.04      0.05    0.35x         0.460
   2        0.10      0.08      0.09      0.16      0.08      0.11    0.67x         0.708
   4        0.20      0.17      0.19      0.23      0.15      0.21    0.93x         0.915
   8        0.39      0.34      0.37      0.40      0.30      0.42    1.03x         0.993
  16        0.78      0.67      0.73      0.78      0.61      0.82    1.04x         1.000
  32        1.50      1.34      1.42      1.49      1.22      1.57    1.05x         1.000
  64        2.80      2.69      2.66      2.74      2.43      2.95    1.05x         1.000
 128        4.98      4.77      4.76      4.77      4.86      5.22    1.05x         1.000
==============================================================================
CPU-only   Qwen2.5 72B FP16  16H+48C   batch sweep
==============================================================================
   B     FlexGen       LIA    AimPod  LLMFlash      ITME TierSched    ratio   active_frac
------------------------------------------------------------------------------
   1        0.06      0.05      0.06      0.31      0.04      0.07    0.21x         0.460
   2        0.12      0.10      0.11      0.22      0.08      0.13    0.60x         0.708
   4        0.24      0.19      0.22      0.28      0.15      0.26    0.92x         0.915
   8        0.48      0.38      0.43      0.49      0.30      0.51    1.04x         0.993
  16        0.94      0.77      0.85      0.94      0.61      0.99    1.06x         1.000
  32        1.80      1.53      1.63      1.78      1.22      1.90    1.06x         1.000
  64        3.32      3.07      3.01      3.21      2.43      3.51    1.06x         1.000
 128        5.77      5.30      5.30      5.37      4.86      6.08    1.05x         1.000
==============================================================================
CPU-only   Qwen2.5 72B FP16  16H+64C   batch sweep
==============================================================================
   B     FlexGen       LIA    AimPod  LLMFlash      ITME TierSched    ratio   active_frac
------------------------------------------------------------------------------
   1        0.08      0.06      0.07      0.40      0.04      0.08    0.21x         0.460
   2        0.15      0.11      0.13      0.35      0.08      0.17    0.48x         0.708
   4        0.31      0.22      0.26      0.37      0.15      0.33    0.90x         0.915
   8        0.61      0.45      0.51      0.62      0.30      0.66    1.06x         0.993
  16        1.18      0.89      1.00      1.18      0.61      1.28    1.08x         1.000
  32        2.24      1.78      1.91      2.20      1.22      2.41    1.08x         1.000
  64        4.06      3.57      3.48      3.88      2.43      4.36    1.07x         1.000
 128        6.87      6.01      5.95      6.27      4.86      7.32    1.06x         1.000
==============================================================================
CPU-only   Qwen2.5 72B INT8  16H+32C   batch sweep
==============================================================================
   B     FlexGen       LIA    AimPod  LLMFlash      ITME TierSched    ratio   active_frac
------------------------------------------------------------------------------
   1        0.21      0.11      0.17      0.96      0.08      0.24    0.25x         0.460
   2        0.43      0.22      0.32      1.36      0.17      0.48    0.35x         0.708
   4        0.84      0.45      0.64      1.12      0.33      0.95    0.84x         0.915
   8        1.66      0.89      1.28      1.72      0.67      1.85    1.07x         0.993
  16        3.19      1.78      2.46      3.20      1.34      3.55    1.11x         1.000
  32        5.96      3.57      4.59      5.75      2.67      6.57    1.10x         1.000
  64       10.50      7.13      8.06      9.57      5.34     11.47    1.09x         1.000
 128       16.96     14.23     13.04     14.34     10.68     18.28    1.08x         1.000
==============================================================================
CPU-only   Qwen2.5 72B INT8  16H+48C   batch sweep
==============================================================================
   B     FlexGen       LIA    AimPod  LLMFlash      ITME TierSched    ratio   active_frac
------------------------------------------------------------------------------
   1        0.45      0.16      0.31      0.85      0.08      0.48    0.56x         0.460
   2        0.89      0.32      0.61      1.25      0.17      0.95    0.76x         0.708
   4        1.77      0.65      1.22      2.01      0.33      1.89    0.94x         0.915
   8        3.52      1.29      2.30      3.63      0.67      3.76    1.03x         0.993
  16        6.92      2.59      4.30      6.86      1.34      7.40    1.07x         1.000
  32       13.40      5.17      7.59     12.54      2.67     14.38    1.07x         1.000
  64       25.01     10.33     12.56     19.92      5.34     27.14    1.09x         1.000
 128       31.94     20.60     18.18     23.39     10.68     36.62    1.15x         1.000
==============================================================================
CPU-only   Qwen2.5 72B INT8  16H+64C   batch sweep
==============================================================================
   B     FlexGen       LIA    AimPod  LLMFlash      ITME TierSched    ratio   active_frac
------------------------------------------------------------------------------
   1        0.44      0.31      0.40      0.84      0.44      0.48    0.57x         0.460
   2        0.89      0.62      0.80      1.23      0.88      0.95    0.77x         0.708
   4        1.77      1.23      1.60      2.00      1.77      1.89    0.94x         0.915
   8        3.50      2.46      3.18      3.73      3.51      3.76    1.01x         0.993
  16        6.88      4.92      6.30      7.30      6.93      7.40    1.01x         1.000
  32       13.30      9.82     12.35     14.18     13.46     14.38    1.01x         1.000
  64       24.93     19.59     23.75     26.75     25.49     27.14    1.01x         1.000
 128       45.84     39.00     29.74     32.75     10.68     48.90    1.07x         1.000
==============================================================================
+RTX 5090 (28 GB)   Qwen2.5 72B FP16  16H+32C   batch sweep
==============================================================================
   B     FlexGen       LIA    AimPod  LLMFlash      ITME TierSched    ratio   active_frac
------------------------------------------------------------------------------
   1        0.07      0.06      0.07      0.76      0.04      0.08    0.10x         0.460
   2        0.15      0.11      0.13      0.32      0.08      0.16    0.49x         0.708
   4        0.29      0.23      0.27      0.36      0.17      0.31    0.88x         0.915
   8        0.57      0.45      0.53      0.60      0.33      0.62    1.02x         0.993
  16        1.11      0.91      1.03      1.16      0.67      1.20    1.04x         1.000
  32        2.12      1.81      1.98      2.19      1.34      2.27    1.04x         1.000
  64        3.87      3.61      3.66      3.94      2.67      4.12    1.05x         1.000
 128        6.60      6.50      6.45      6.60      5.35      6.96    1.05x         1.000
==============================================================================
+RTX 5090 (28 GB)   Qwen2.5 72B FP16  16H+48C   batch sweep
==============================================================================
   B     FlexGen       LIA    AimPod  LLMFlash      ITME TierSched    ratio   active_frac
------------------------------------------------------------------------------
   1        0.10      0.07      0.08      0.61      0.04      0.11    0.18x         0.460
   2        0.20      0.14      0.17      0.73      0.08      0.22    0.30x         0.708
   4        0.39      0.27      0.33      0.51      0.17      0.43    0.84x         0.915
   8        0.76      0.54      0.65      0.81      0.33      0.85    1.04x         0.993
  16        1.48      1.09      1.27      1.53      0.67      1.63    1.06x         1.000
  32        2.77      2.17      2.41      2.85      1.34      3.03    1.07x         1.000
  64        4.92      4.33      4.37      4.98      2.67      5.33    1.07x         1.000
 128        8.07      7.53      7.49      7.97      5.35      8.59    1.07x         1.000
==============================================================================
+RTX 5090 (28 GB)   Qwen2.5 72B FP16  16H+64C   batch sweep
==============================================================================
   B     FlexGen       LIA    AimPod  LLMFlash      ITME TierSched    ratio   active_frac
------------------------------------------------------------------------------
   1        0.15      0.08      0.11      0.54      0.04      0.18    0.33x         0.460
   2        0.30      0.17      0.21      0.79      0.08      0.35    0.45x         0.708
   4        0.59      0.34      0.43      0.92      0.17      0.70    0.76x         0.915
   8        1.15      0.68      0.85      1.25      0.33      1.35    1.08x         0.993
  16        2.19      1.36      1.65      2.29      0.67      2.55    1.11x         1.000
  32        3.99      2.71      3.07      4.11      1.34      4.56    1.11x         1.000
  64        6.77      5.40      5.43      6.80      2.67      7.55    1.11x         1.000
 128       10.38      9.05      8.84     10.10      5.35     11.28    1.09x         1.000
==============================================================================
+RTX 5090 (28 GB)   Qwen2.5 72B INT8  16H+32C   batch sweep
==============================================================================
   B     FlexGen       LIA    AimPod  LLMFlash      ITME TierSched    ratio   active_frac
------------------------------------------------------------------------------
   1        0.96      0.35      0.75      1.82      0.08      1.06    0.58x         0.460
   2        1.90      0.70      1.49      2.66      0.17      2.10    0.79x         0.708
   4        3.77      1.41      2.97      4.33      0.33      4.17    0.96x         0.915
   8        7.40      2.82      5.92      7.99      0.67      8.21    1.03x         0.993
  16       14.29      5.63     11.66     15.39      1.34     15.91    1.03x         1.000
  32       26.70     11.19     22.56     28.97      2.67     29.96    1.03x         1.000
  64       47.22     22.09     40.16     51.84      5.35     53.53    1.03x         1.000
 128       76.51     43.04     46.29     75.12     10.69     87.88    1.15x         1.000
==============================================================================
+RTX 5090 (28 GB)   Qwen2.5 72B INT8  16H+48C   batch sweep
==============================================================================
   B     FlexGen       LIA    AimPod  LLMFlash      ITME TierSched    ratio   active_frac
------------------------------------------------------------------------------
   1        0.96      0.61      0.75      1.82      0.08      1.06    0.58x         0.460
   2        1.90      1.22      1.49      2.66      0.17      2.10    0.79x         0.708
   4        3.77      2.44      2.97      4.33      0.33      4.17    0.96x         0.915
   8        7.40      4.88      5.92      7.99      0.67      8.21    1.03x         0.993
  16       14.29      9.75     11.66     15.39      1.34     15.91    1.03x         1.000
  32       26.70     19.28     22.56     28.97      2.67     29.96    1.03x         1.000
  64       47.19     37.71     42.32     51.86      5.35     53.53    1.03x         1.000
 128       76.60     72.20     75.29     85.65     10.69     87.88    1.03x         1.000
==============================================================================
+RTX 5090 (28 GB)   Qwen2.5 72B INT8  16H+64C   batch sweep
==============================================================================
   B     FlexGen       LIA    AimPod  LLMFlash      ITME TierSched    ratio   active_frac
------------------------------------------------------------------------------
   1        0.96      0.61      0.75      1.82      0.44      1.06    0.58x         0.460
   2        1.90      1.22      1.49      2.66      0.88      2.10    0.79x         0.708
   4        3.77      2.44      2.97      4.33      1.77      4.17    0.96x         0.915
   8        7.40      4.88      5.92      7.99      3.53      8.21    1.03x         0.993
  16       14.29      9.75     11.66     15.39      7.06     15.91    1.03x         1.000
  32       26.70     19.28     22.56     28.97     14.13     29.96    1.03x         1.000
  64       47.19     37.71     42.32     51.86     28.25     53.53    1.03x         1.000
 128       76.60     72.20     75.29     85.71     56.48     88.04    1.03x         1.000
```

## Fig. 6: model scale

`python run_paper_tables.py --models`

```
==============================================================================
Model scale, FP16 16H+48C, CPU-only, B=128
==============================================================================
Model           FlexGen       LIA    AimPod  LLMFlash      ITME TierSched    ratio
------------------------------------------------------------------------------
Mistral 7B       231.45    188.64    299.46    304.48    278.17    321.19    1.05x
Llama 13B         21.28     21.69     22.37     16.93     20.37     22.88    1.02x
Qwen3 22B         75.44     75.84     76.36     79.78     73.49     81.94    1.03x
Qwen2.5 72B        5.77      5.30      5.30      5.37      4.86      6.08    1.05x
Llama 405B         0.75      0.76      0.76      0.75      0.78      0.79    1.02x
==============================================================================
Model scale, INT8 16H+48C, CPU-only, B=128
==============================================================================
Model           FlexGen       LIA    AimPod  LLMFlash      ITME TierSched    ratio
------------------------------------------------------------------------------
Mistral 7B       278.85    318.82    594.58    382.46    574.55    628.89    1.06x
Llama 13B         91.47     91.58     97.56    111.68    102.63    110.13    0.99x
Qwen3 22B        205.59    156.32    210.79    269.78    200.10    274.54    1.02x
Qwen2.5 72B       31.94     20.60     18.18     23.39     10.68     36.62    1.15x
Llama 405B         1.56      1.55      1.55      1.57      1.60      1.74    1.09x
==============================================================================
Model scale, FP16 16H+48C, +RTX 5090 (28 GB), B=128
==============================================================================
Model           FlexGen       LIA    AimPod  LLMFlash      ITME TierSched    ratio
------------------------------------------------------------------------------
Mistral 7B      4904.06    426.81    305.91   7635.01    291.18   9193.10    1.20x
Llama 13B         64.82     62.17     34.31     64.46     55.38     64.32    0.99x
Qwen3 22B        192.81    169.30    177.04    276.88    101.14    277.42    1.00x
Qwen2.5 72B        8.07      7.53      7.49      7.97      5.35      8.59    1.07x
Llama 405B         0.81      0.82      0.81      0.82      0.80      0.83    1.01x
==============================================================================
Model scale, INT8 16H+48C, +RTX 5090 (28 GB), B=128
==============================================================================
Model           FlexGen       LIA    AimPod  LLMFlash      ITME TierSched    ratio
------------------------------------------------------------------------------
Mistral 7B      6689.52    820.33    611.27   9348.17    581.95  10470.85    1.12x
Llama 13B        314.79    184.72     97.94    318.71    379.99    410.14    1.08x
Qwen3 22B       2323.20    591.94    408.71   3579.34    202.16   3694.25    1.03x
Qwen2.5 72B       76.60     72.20     75.29     85.65     10.69     87.88    1.03x
Llama 405B         1.85      1.82      1.82      1.83      1.66      1.90    1.03x
```

## Realistic host

`python realistic_runs.py host`

```
Realistic host: 161 GB/s, 48 GB device cache, B=128, decode 256
plat model quant   host           best baseline TierSched   ratio
CPU  72B   fp16    256G      101.190 (AimPod  )   102.089   1.01x
CPU  72B   fp16    512G      101.190 (AimPod  )   102.025   1.01x
CPU  72B   int8    256G      101.729 (AimPod  )   102.248   1.01x
CPU  72B   int8    512G      101.729 (AimPod  )   102.248   1.01x
CPU  405B  fp16    256G        1.099 (LLMFlash)     1.182   1.08x
CPU  405B  fp16    512G        2.135 (LLMFlash)     2.472   1.16x
CPU  405B  int8    256G        4.535 (LLMFlash)     6.212   1.37x
CPU  405B  int8    512G       17.868 (LLMFlash)    17.877   1.00x
GPU  72B   fp16    256G      130.951 (LLMFlash)   150.324   1.15x
GPU  72B   fp16    512G      130.951 (LLMFlash)   131.379   1.00x
GPU  72B   int8    256G      332.115 (LLMFlash)   334.794   1.01x
GPU  72B   int8    512G      332.116 (LLMFlash)   334.794   1.01x
GPU  405B  fp16    256G        1.234 (LLMFlash)     1.254   1.02x
GPU  405B  fp16    512G        2.710 (LLMFlash)     2.807   1.04x
GPU  405B  int8    256G        8.245 (LLMFlash)     8.879   1.08x
GPU  405B  int8    512G       48.229 (AimPod  )    50.184   1.04x
```

## ShareGPT prompts

`python run_paper_tables.py --sharegpt`

```
ShareGPT replay, 50 prompts, FP16 16H+64C, CPU-only, B=128
prefill: median 89  P90 955  max 1814
  win rate           50/50 (100%)
  median speedup     1.08x
  P10 / P90          1.06x / 1.08x
  min / max          1.05x / 1.08x
ShareGPT replay, 50 prompts, FP16 16H+64C, +RTX 5090, B=128
prefill: median 89  P90 955  max 1814
  win rate           50/50 (100%)
  median speedup     1.11x
  P10 / P90          1.07x / 1.12x
  min / max          1.02x / 1.12x
```

## Re-planning cost

`python replan_cost.py`

```
worst re-plan cost as a share of one generation: 6.64 % ('int8', 64, 28, 512, 256, 21.2, 320)
```

## CMM-H's default LRU cache against TierSched, and Belady's rule against lookahead

`DYN_POLICY=lru python dynamic_baseline.py (in a configured tree; see run_paper_tables._run_variant)`

```
CPU fp16 16H+32C  LRU 4.84  TierSched 5.22  ratio 1.08
CPU fp16 16H+48C  LRU 4.93  TierSched 6.08  ratio 1.23
CPU fp16 16H+64C  LRU 4.94  TierSched 7.32  ratio 1.48
CPU int8 16H+32C  LRU 11.35  TierSched 18.28  ratio 1.61
CPU int8 16H+48C  LRU 11.35  TierSched 36.62  ratio 3.23
CPU int8 16H+64C  LRU 46.55  TierSched 48.90  ratio 1.05
GPU fp16 16H+32C  LRU 6.26  TierSched 6.96  ratio 1.11
GPU fp16 16H+48C  LRU 6.39  TierSched 8.59  ratio 1.34
GPU fp16 16H+64C  LRU 6.42  TierSched 11.28  ratio 1.76
GPU int8 16H+32C  LRU 24.34  TierSched 87.88  ratio 3.61
GPU int8 16H+48C  LRU 87.02  TierSched 87.88  ratio 1.01
GPU int8 16H+64C  LRU 87.02  TierSched 88.04  ratio 1.01
Belady CPU INT8 48C K= 1 21.778199
Belady CPU INT8 48C K= 4 23.099641
Belady CPU INT8 48C K= 8 24.009391
Belady CPU INT8 48C K= 16 26.061333
Belady CPU INT8 48C K= 32 22.63923
Belady CPU INT8 48C K= 4096 11.442048
```

## Search time per configuration (one run at a time)

`python tiersched.py (Search time line)`

```
CPU fp16 16H+32C 4.60 s (36 finalists)
CPU fp16 16H+48C 9.42 s (72 finalists)
CPU fp16 16H+64C 14.88 s (108 finalists)
CPU int8 16H+32C 5.54 s (36 finalists)
CPU int8 16H+48C 11.99 s (72 finalists)
CPU int8 16H+64C 17.70 s (108 finalists)
GPU fp16 16H+32C 7.28 s (48 finalists)
GPU fp16 16H+48C 14.62 s (96 finalists)
GPU fp16 16H+64C 21.87 s (144 finalists)
GPU int8 16H+32C 8.06 s (48 finalists)
GPU int8 16H+48C 14.20 s (96 finalists)
GPU int8 16H+64C 19.95 s (144 finalists)
```

## Nine-step finalist timing against timing every step

`set _probe = list(range(TOKENS)) in tiersched.py`

```
CPU fp16 32C  nine-step 5.218  every-step 5.218  ratio 1.0000  DIFFERENT: device 32 GB of 32, KV tier stripe reserved at mean, layout stripe, reserves host 0.00 GB device 0.48 GB | device 32 GB of 32, KV tier stripe reserved at mean, layout stripe, reserves host 0.48 GB device 0.00 GB
CPU fp16 48C  nine-step 6.079  every-step 6.079  ratio 1.0000  same plan
CPU fp16 64C  nine-step 7.316  every-step 7.316  ratio 1.0000  same plan
CPU int8 32C  nine-step 18.282  every-step 18.282  ratio 1.0000  DIFFERENT: device 32 GB of 32, KV tier stripe reserved at mean, layout stripe, reserves host 0.24 GB device 0.00 GB | device 32 GB of 32, KV tier stripe reserved at mean, layout stripe, reserves host 0.00 GB device 0.24 GB
CPU int8 48C  nine-step 36.617  every-step 36.617  ratio 1.0000  same plan
CPU int8 64C  nine-step 48.904  every-step 48.904  ratio 1.0000  same plan
GPU fp16 32C  nine-step 6.961  every-step 6.961  ratio 1.0000  same plan
GPU fp16 48C  nine-step 8.593  every-step 8.593  ratio 1.0000  same plan
GPU fp16 64C  nine-step 11.275  every-step 11.275  ratio 1.0000  same plan
GPU int8 32C  nine-step 87.881  every-step 87.881  ratio 1.0000  same plan
GPU int8 48C  nine-step 87.881  every-step 87.881  ratio 1.0000  same plan
GPU int8 64C  nine-step 88.041  every-step 88.041  ratio 1.0000  same plan
identical throughput in 12/12
```

## Correctness invariants

`python verify_results.py`

```
  WARN  ITME      int8: 581.95 -> 202.16 -> 379.99 -> 10.69 -> 1.66  MORE BYTES, FASTER: [('Qwen3_20BCfg', 202.160381, 'Llama13BCfg', 379.987164)]
211 passed, 1 warnings, 0 FAILED
```
