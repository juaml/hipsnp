#import ..hipsnp as hps
print('__file__={0:<35} | __name__={1:<25} | __package__={2:<25}'.format(__file__,__name__,str(__package__)))
import hipsnp
import pandas as pd
import datalad.api as dl

def test_datalad_get_chromosome_return_DataladType():
    """Read example data from GIN, assert returnst datalad object"""
    c = '1'
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen' # exmaple data on GIN
    
    _,dataladObject,_ = hipsnp.datalad_get_chromosome(c=1,source=source)
    # assert type(dataladObject) == dl.Dataset
    print(type(dataladObject) == dl.Dataset)

if __name__ == "__main__":
    test_datalad_get_chromosome_return_DataladType()