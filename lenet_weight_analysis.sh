
# rm -rf /Users/sahib/open_lth_data2/*
# python open_lth.py train --default_hparams=mnist_lenet_300_100

python open_lth.py branch train oneshot --default_hparams=mnist_lenet_300_100 --strategy=magnitude --prune_fraction=0.75
python open_lth.py branch train oneshot --default_hparams=mnist_lenet_300_100 --strategy=magnitude --prune_fraction=0.75 --reinitialize
python open_lth.py branch train oneshot --default_hparams=mnist_lenet_300_100 --strategy=magnitude --prune_fraction=0.75 --randomize_layerwise


