#PBS -lwalltime=24:00:00
#PBS -lselect=1:ncpus=1:mem=2gb
#PBS -J 1-50000

module load anaconda3/personal
source ~/anaconda3/bin/activate fastms

export MSDIR=$HOME/fastms
export OUTPUT=ibm_priors_$PBS_ARRAY_INDEX

python -m fastms sample ibm prior $OUTPUT --sites data -n 10 --cores 1 \
  --burnin 50 --start 1985 --end 2018 --seed=$PBS_ARRAY_INDEX
