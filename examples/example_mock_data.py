"""
Obtain risk scores and alleles
==============================

 This example uses a mock bgen file with genotic information hosted in GIN.
 Basic allele and risk score information is computed 
 .. include:: ../../links.inc
"""
# # Authors: Oscar Portoles <o.portoles@fz-juelich.de>
#            Federico Raimondo <f.raimondo@fz-juelich.de>
#
# License: AGPL

import datalad.api as dl
import hipsnp as hps
from hipsnp.utils import configure_logging

###############################################################################
# Set the logging level to info to see extra information
configure_logging(level='INFO')

###############################################################################
# Obtain .bgen file from a remote source as a Datalad dataset
dl_source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'  # exmaple data
path_to_save_bgen = '/tmp/hipsnp/example1A/'

dataset = dl.clone(source=dl_source, path=path_to_save_bgen)
dataset.get()

###############################################################################
# Next we will read the downladed .bgen file
# path to the downloaded .bgen file
bgenfile = path_to_save_bgen + 'imputation/example_c1_v0.bgen'
gen = hps.Genotype.from_bgen(files=bgenfile)

# Now we are ready to obtain the alleles of each rsid and smaple in the data
gen_allele, gen_012 = gen.alleles()

# For example, we can count the number of times that an allele appears
gen_allele.loc['RSID_2'].value_counts()

#############################################################################
# Obtain poligenetic risk score we need a file or a pandas data frame with
# the poligenetic risk score associated to each rsid. We can search on and 
# retrieve from https://www.pgscatalog.org/ traits and poligentic risk scores.

# In this case we will download a .csv file with mock risk score for the the 
# rsids in the previous file

path_to_weights = './data/weights_all.csv'

# Now we can obtain the risk score for each rsids given the samples on the 
# dataset and the dosage (amount of the effect allele) of each rsids and sample
dosage, risk = gen.riskscore(weights=path_to_weights)

# Then, for example, we can visualize the risk socre of each sample
risk.plot()
