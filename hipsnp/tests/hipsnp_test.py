import pytest
import os
import hipsnp as hps
import json
import pandas as pd
import datalad.api as dl


mock_rsid = 'rs699'

def validateJSON(jsonData):
    """attemts to open a JSON var"""
    try:
        json_object = json.loads(jsonData)
    except ValueError as e:
        return False
    return True


def test_JSON_format_ensembl_human_rsid():
    """test output is in JSON format"""
    mock_rsid = 'rs699'
    outJSON = hps.ensembl_human_rsid(mock_rsid)
    assert validateJSON(outJSON)


def validatePANDAStype(pdData):
    """Data is a pandas DataFrame with fields 'chromosomes' and 'rsids' of type str"""
    assert isinstance(pdData, pd.core.frame.DataFrame)

def test_rsid2chromosome_has_pandas_format():
    mock_rsid = 'rs699'
    outPANDAS = hps.rsid2chromosome(mock_rsid)
    validatePANDAStype(outPANDAS)

def valiadtePANDAS_has_RSDIandCROMOSOM(pdData, refColFields):
    outFields = [field for field in pdData.columns]
    assert refColFields.sort() == outFields.sort()

def test_rsid2chromosome_has_RSIDandCROMOSOM():
    mock_rsid = 'rs699'
    refColFields = ['rsids','chromosomes']
    outPANDAS = hps.rsid2chromosome(mock_rsid)

    valiadtePANDAS_has_RSDIandCROMOSOM(outPANDAS, refColFields)


def test_rsid2chromosome_has_list_of_RSIDandCROMOSOM():
    mock_rsid = ['rs699', 'rs698']
    refColFields = ['rsids','chromosomes']
    outPANDAS = hps.rsid2chromosome(mock_rsid)

    valiadtePANDAS_has_RSDIandCROMOSOM(outPANDAS, refColFields)

def removeDATALADdataset(dlObject):
    dlObject.remove()

def test_datalad_get_chromosome_return_DataladType():
    """Read example data from GIN, assert returnst datalad object"""
    c = '1'
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen' # exmaple data on GIN
    
    _,dataladObject,_ = hps.datalad_get_chromosome(c=1,source=source)
    assert type(dataladObject) == dl.Dataset
    removeDATALADdataset(dataladObject)

def validateNameObtainedFiles(dataLget):
    """files obtined with DataLad are the exnple files"""
    filenames = [os.path.basename(ind['path']) for ind in dataLget if ind['type'] == 'file']
    sameFiles = 'example_c1_v0.bgen' and 'example_c1_v0.sample' in filenames
    return sameFiles

def test_datalad_get_chromosome_dataland_get():
    """ dataland.get()_returns_expected_data structure """
    c = '1'
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen' # exmaple data on GIN

    _,dataladObject,dataGet = hps.datalad_get_chromosome(c=1,source=source)
    assert validateNameObtainedFiles(dataGet)
    removeDATALADdataset(dataladObject)

@pytest.mark.parametrize("rsids, qctool",
                        [('101', '/home/oportoles/Apps/qctool_v2.0.6-Ubuntu16.04-x86_64/qctool'),
                        ('101', None),
                        ('101', '/home/oportoles/Apps'),
                        ('101', '/home/oportoles/Apps/qctool_v2.0.6-Ubuntu16.04-x86_64')])
def test_rsid2vcf_qctool(rsids, qctool):
# def test_rsid2vcf_qctool():
    """ finds and uses qctool"""
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen' # exmaple data on GIN
    outdir = '/home/oportoles/Documents/MyCode/hipsnp/test_data'
    rsids = 'rs101'

    # qctool = '/home/oportoles/Apps/qctool_v2.0.6-Ubuntu16.04-x86_64'
    
    # returns: a pandas dataframe with rsid-chromosome pairs
    #  and the vcf files are created in the outdir
    # ch_rs: pandas DataFrame 'chromosomes' 'rsids'
    # files: 
    # ds: datalad DS
    ch_rs, _, dataL = hps.rsid2vcf(rsids, outdir,        
                                    datalad_source=source,
                                    qctool=qctool,
                                    datalad_drop=True,
                                    datalad_drop_if_got=True,
                                    datalad_dir=None,
                                    force=False,
                                    chromosomes=None,
                                    chromosomes_use=None)
    
    assert type(dataL) == dl.Dataset 
    removeDATALADdataset(dataL)




