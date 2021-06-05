
# rm -rf /Users/sahib/open_lth_data2/*
# python open_lth.py train --default_hparams=cifar_resnet_20

python open_lth.py branch train oneshot --default_hparams=cifar_resnet_20 --strategy=magnitude --prune_fraction=0.75
python open_lth.py branch train oneshot --default_hparams=cifar_resnet_20 --strategy=magnitude --prune_fraction=0.75 --reinitialize
python open_lth.py branch train oneshot --default_hparams=cifar_resnet_20 --strategy=magnitude --prune_fraction=0.75 --randomize_layerwise
python open_lth.py branch train oneshot --default_hparams=cifar_resnet_20 --strategy=random --prune_fraction=0.75