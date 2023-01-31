#!/bin/bash

#Random
python open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=random --prune_fraction=0.0
python open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=random --prune_fraction=0.5
python open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=random --prune_fraction=0.75
python open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=random --prune_fraction=0.875
python open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=random --prune_fraction=0.9375
python open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=random --prune_fraction=0.96875
python open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=random --prune_fraction=0.98438
python open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=random --prune_fraction=0.99219
python open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=random --prune_fraction=0.99609

#LTH
python open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=magnitude --prune_fraction=0.0
python open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=magnitude --prune_fraction=0.5
python open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=magnitude --prune_fraction=0.75
python open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=magnitude --prune_fraction=0.875
python open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=magnitude --prune_fraction=0.9375
python open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=magnitude --prune_fraction=0.96875
python open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=magnitude --prune_fraction=0.98438
python open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=magnitude --prune_fraction=0.99219
python open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=magnitude --prune_fraction=0.99609

#Synflow
python open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=synflow --prune_fraction=0.0 --prune_iterations=100
python open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=synflow --prune_fraction=0.5 --prune_iterations=100
python open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=synflow --prune_fraction=0.75 --prune_iterations=100
python open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=synflow --prune_fraction=0.875 --prune_iterations=100
python open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=synflow --prune_fraction=0.9375 --prune_iterations=100
python open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=synflow --prune_fraction=0.96875 --prune_iterations=100
python open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=synflow --prune_fraction=0.98438 --prune_iterations=100
python open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=synflow --prune_fraction=0.99219 --prune_iterations=100
python open_lth.py branch train oneshot --default_hparams=cifar_vgg_16 --strategy=synflow --prune_fraction=0.99609 --prune_iterations=100