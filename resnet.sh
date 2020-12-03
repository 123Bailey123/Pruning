

python open_lth.py train --default_hparams=cifar_resnet_20

python open_lth.py branch train oneshot --default_hparams=cifar_resnet_20 --strategy=magnitude --prune_fraction=0.75
python open_lth.py branch train oneshot --default_hparams=cifar_resnet_20 --strategy=random --prune_fraction=0.75
python open_lth.py branch train oneshot --default_hparams=cifar_resnet_20 --strategy=snip10 --prune_fraction=0.75
python open_lth.py branch train oneshot --default_hparams=cifar_resnet_20 --strategy=grasp10 --prune_fraction=0.75
python open_lth.py branch train oneshot --default_hparams=cifar_resnet_20 --strategy=graspabs10 --prune_fraction=0.75
python open_lth.py branch train oneshot --default_hparams=cifar_resnet_20 --strategy=synflow --prune_fraction=0.75 --prune_iterations=100
python -c 'from merge import pdf_merge; pdf_merge("Data_Distribution/")'

python open_lth.py branch train oneshot --default_hparams=cifar_resnet_20 --strategy=magnitude --prune_fraction=0.75 --reinitialize
python open_lth.py branch train oneshot --default_hparams=cifar_resnet_20 --strategy=random --prune_fraction=0.75 --reinitialize
python open_lth.py branch train oneshot --default_hparams=cifar_resnet_20 --strategy=snip10 --prune_fraction=0.75 --reinitialize
python open_lth.py branch train oneshot --default_hparams=cifar_resnet_20 --strategy=grasp10 --prune_fraction=0.75 --reinitialize
python open_lth.py branch train oneshot --default_hparams=cifar_resnet_20 --strategy=graspabs10 --prune_fraction=0.75 --reinitialize
python open_lth.py branch train oneshot --default_hparams=cifar_resnet_20 --strategy=synflow --prune_fraction=0.75 --prune_iterations=100 --reinitialize
python -c 'from merge import pdf_merge; pdf_merge("Data_Distribution_Reinit/")' 

python open_lth.py branch train oneshot --default_hparams=cifar_resnet_20 --strategy=magnitude --prune_fraction=0.75 --randomize_layerwise
python open_lth.py branch train oneshot --default_hparams=cifar_resnet_20 --strategy=random --prune_fraction=0.75 --randomize_layerwise
python open_lth.py branch train oneshot --default_hparams=cifar_resnet_20 --strategy=snip10 --prune_fraction=0.75 --randomize_layerwise
python open_lth.py branch train oneshot --default_hparams=cifar_resnet_20 --strategy=grasp10 --prune_fraction=0.75 --randomize_layerwise
python open_lth.py branch train oneshot --default_hparams=cifar_resnet_20 --strategy=graspabs10 --prune_fraction=0.75 --randomize_layerwise
python open_lth.py branch train oneshot --default_hparams=cifar_resnet_20 --strategy=synflow --prune_fraction=0.75 --prune_iterations=100 --randomize_layerwise
python -c 'from merge import pdf_merge; pdf_merge("Data_Distribution__Randomize_Layerwise/")' 


