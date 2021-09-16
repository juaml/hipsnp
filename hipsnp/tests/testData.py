from sys import path
from bgen_reader import example_filepath
from bgen_reader import read_bgen
from datalad import api as dl
import numpy as np

filepath = {}
filepath["example.bgen"] = example_filepath("example.bgen")
#filepath["haplotypes.bgen"] = example_filepath("haplotypes.bgen")
#filepath["complex.bgen"] = example_filepath("complex.bgen")


#bgen = read_bgen(filepath["example.bgen"], verbose=False)




#data = dl.Dataset()
#print(type(data))
#dl.create(path='./hispnp/test/testBGEN', cfg_proc='text2git')

#dl.install(source='https://github.com/datalad-datasets/longnow-podcasts.git',
#          get_data=True)

