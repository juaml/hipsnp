"""
Obtain risk scores and alleles
==============================

This example uses a mock dataset file with genotic information hosted in GIN.
The alleles are extracted from the BGEN files and a risk score is calculated

.. include:: ../links.inc
"""
# # Authors: Oscar Portoles <o.portoles@fz-juelich.de>
#            Federico Raimondo <f.raimondo@fz-juelich.de>
#
# License: AGPL

from tempfile import mkdtemp
import shutil
from hipsnp.hipsnp import genotype_from_datalad
from hipsnp.utils import configure_logging

###############################################################################
# Set the logging level to info to see extra information
configure_logging(level='INFO')

###############################################################################
# Get a temporary directory to work. Use another directory if you want to
# keep the data.
workdir = mkdtemp()

###############################################################################
# Obtain a genotype from a datalad dataset. We can use the same directory
# as work and data directory.
source = 'https://gin.g-node.org/juaml/datalad-example-bgen'
genotype = genotype_from_datalad(
    rsids=['RSID_1', 'RSID_2', 'RSID_3'], chromosomes=['1', '1', '1'],
    datalad_source=source, workdir=workdir, datadir=workdir,)

###############################################################################
# Now we are ready to obtain the alleles of each rsid and sample in the data
gen_allele, gen_012 = genotype.alleles()

# For example, we can count the number of times that an allele appears
gen_allele.loc['RSID_2'].value_counts()

##############################################################################
# To compute a poligenetic risk score we need a file or a pandas dataframe with
# the weights associated to each allele and RSID. We can retrieve one from
# https://www.pgscatalog.org/.

# In this case we use a weights file generated for this mock datasets.
path_to_weights = './data/weights_all.csv'

# Now we can obtain the risk score for each rsids given the samples on the
# dataset and the dosage (amount of the effect allele) of each rsids and sample
dosage, risk = genotype.riskscores(weights=path_to_weights)

##############################################################################
# Then, for example, we can visualize the risk score of each sample
risk.plot(ls='none', marker='*', legend=False)

###############################################################################
# Since we used a temporary directory, we need to delete it
shutil.rmtree(workdir)
