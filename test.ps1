param(
    [int]$GPUID
)

$OUTDIR = "outputs/permuted_MNIST_incremental_domain"
$REPEAT = 10

# 创建目录，如果目录不存在的话
if (!(Test-Path -Path $OUTDIR)) {
    New-Item -ItemType Directory -Force -Path $OUTDIR
}

# 运行Python脚本
python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer SGD  --n_permutation 10 --no_class_remap --force_out_dim 10 --schedule 10 --batch_size 128 --model_name MLP1000 --agent_type customization  --agent_name GEM_16000  --lr 0.1 --reg_coef 0.5