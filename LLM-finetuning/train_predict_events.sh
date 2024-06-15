#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=00:50:00
#SBATCH --output=globalise/train_predict_events.out

# Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0

# activate virtual environment
source venv22/bin/activate
# pip list

#Create output directory on scratch
mkdir -p "$TMPDIR"/data

#Copy data to scratch
cp $HOME/globalise/data/train.json "$TMPDIR"/data
cp $HOME/globalise/data/dev.json "$TMPDIR"/data
cp $HOME/globalise/data/test.json "$TMPDIR"/data
cp -r globalise/cfg "$TMPDIR"/cfg

#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.

python run_event_detect.py fit -c globalise/cfg/train.yaml -c globalise/cfg/glb_events.yaml --data.data_dir=globalise/data \
	--trainer.default_root_dir="$TMPDIR"/experiment

python run_event_detect.py predict -c "$TMPDIR"/experiment/config.yaml -c globalise/cfg/predict.yaml \
	--data.data_dir="$TMPDIR"/data --trainer.default_root_dir="$TMPDIR"/experiment \
	--ckpt_path="$TMPDIR"/experiment/lightning_logs/version_0/checkpoints/last.ckpt

#Copy output directory from scratch to home
ls -l "$TMPDIR"/experiment/
outdir="$HOME/ner-finetuning/results/glb_events"
[[ ! -d $outdir ]] && mkdir -p $outdir
cp -r "$TMPDIR"/experiment/lightning_logs $outdir
cp "$TMPDIR"/experiment/config.yaml $outdir
cp "$TMPDIR"/experiment/predictions.json $outdir
cp "$TMPDIR"/experiment/predictions.pt $outdir
cp "$TMPDIR"/predictions.json $outdir
cp "$TMPDIR"/predictions.pt $outdir
