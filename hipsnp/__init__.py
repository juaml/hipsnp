from . _version import __version__
from . hipsnp import (read_bgen, Genotype, read_weights,
                      get_chromosomes_from_ensembl, get_chromosomes,
                      genotype_from_datalad)
from . import utils