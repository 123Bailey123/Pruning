# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from platforms import local
_PLATFORM = local.Platform(num_workers=4)


def get_platform():
    return _PLATFORM
