#import ..hipsnp as hps
import hipsnp
import pandas as pd
import datalad.api as dl



def get_series_of_chromosome_from_RSIDs():
    """get chromosom from user given RSID"""
    rsids = ['rs101', 'rs102', 'rs103', 'rs1', 'rs2','rs3']
    rsids = ['rs_2','rs_3']
    ### copied from rsid2vcf
    for rsid in rsids:
        ch_rs = hipsnp.rsid2chromosome(rsid, chromosomes=None)
        chromosomes = ch_rs['chromosomes'].tolist()
        uchromosomes = pd.unique(chromosomes)
        print('chromosomes needed: ' + str(uchromosomes) + '\n')

def test_datalad_get_chromosome_return_DataladType():
    """Read example data from GIN, assert returnst datalad object"""
    c = '1'
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen' # exmaple data on GIN
    
    _,dataladObject,_ = hipsnp.datalad_get_chromosome(c=1,source=source)
    # assert type(dataladObject) == dl.Dataset
    print(type(dataladObject) == dl.Dataset)

if __name__ == "__main__":
    # test_datalad_get_chromosome_return_DataladType()
    get_series_of_chromosome_from_RSIDs()