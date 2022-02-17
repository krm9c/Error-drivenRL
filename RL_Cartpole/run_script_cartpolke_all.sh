#!/bin/bash

export http_proxy=http://proxy.tmi.alcf.anl.gov:3128
export https_proxy=http://proxy.tmi.alcf.anl.gov:3128


# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/lus/theta-fs0/software/tshetagpu/conda/2021-06-26/mconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/etc/profile.d/conda.sh" ]; then
        . "/lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/etc/profile.d/conda.sh"
    else
        export PATH="/lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<


conda activate torchRL

python cartpole_dual.py --version single_grad --param all
python cartpole_dual.py --version double_grad --param all
python cartpole_dual.py --version single_EDL --param all
python cartpole_dual.py --version double_EDL --param all
conda deactivate
