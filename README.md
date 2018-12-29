# AmygNet

Usage:

1. Download the "NiftyNet" via https://github.com/NifTK/NiftyNet.git
2. Add the "AmygNet.py" file to the NiftyNet/niftynet/network
3. Replace the "Application_factory.py" with my "Application_factory.py" in NiftyNet/niftynet/engine.
4. Follow the instructions on https://niftynet.readthedocs.io/en/dev/config_spec.html for training and testing.

To do the MC testing,
1. choose a dropout rate (set the keep_prob to any number other than 1).
2. python MC_run.sh

Metrics
use the metrics.py
