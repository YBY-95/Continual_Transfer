param(  
    [int]$GPUID = 0,  
    [string]$OUTDIR = "outputs/permuted_MNIST_incremental_class",  
    [int]$REPEAT = 10  
)  
  
# 创建输出目录  
New-Item -ItemType Directory -Force -Path $OUTDIR -ErrorAction SilentlyContinue  

# 定义Python命令的基本部分  
$pythonCommand = "python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --n_permutation 10 --force_out_dim 100 --schedule 10 --batch_size 128 --model_name MLP1000"  
  
# 运行各个Python命令并将输出重定向到日志文件  
Invoke-Expression "$pythonCommand --optimizer Adam    --lr 0.0001  --offline_training"  | Out-File -FilePath "${OUTDIR}/Offline.log" -Append