
## Copyright (C) 2019, Guangke Chen <gkchen.shanghaitech@gmail.com>.
## This program is licenced under the BSD 2-Clause licence
## contained in the LICENCE file in this directory.


spk_ids="1580 2830 4446 5142 61"

archi=gmm
#archi=iv
#task=OSI
# task=CSI
task=SV
attack_type=targeted
# attack_type=untargeted
#threshold=0.144100 # gmm SV threshold value is directly drawn from the running result of "test.py"
#threshold=1.824554 # iv SV
#threshold=2.092623 # iv OSI
threshold=0.227700 # gmm OSI


adver_thresh=0.0
epsilon=0.002
max_iter=1000
offset_iters=0

max_lr=0.001
min_lr=1e-6
#samples=50
samples=50
sigma=0.001
momentum=0.9
plateau_length=5
plateau_drop=2.0

n_jobs=5
debug=f # "f" for False, "t" for True
#debug=t


set -x

spk_ids_list="1580 2830 4446 5142 61"


for mode in ' ' '--offset_training'
do
    for spk_id in $spk_ids_list
    do
        spk_ids=$spk_id
        unbuffer python3 attackMain.py -spk_id $spk_ids -archi $archi -task $task -type $attack_type \
        -adver $adver_thresh -epsilon $epsilon -max_iter $max_iter -max_lr $max_lr \
        -min_lr $min_lr -samples $samples -sigma $sigma -momentum $momentum \
        -plateau_length $plateau_length -plateau_drop $plateau_drop \
        -nj $n_jobs -debug $debug \
        -thresh $threshold --offset_iters $offset_iters \
        $mode
    done | tee -a log
done



