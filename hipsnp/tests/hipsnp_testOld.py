import pytest
import os
from requests.models import Response
import hipsnp as hps
import json
import pandas as pd
import datalad.api as dl
from pathlib import Path, PosixPath
import tempfile


mock_rsid = 'rs699'


@pytest.fixture
def clean_datalad_Datasets(request):
    ''' check for and remove datalad datasets'''
    test_data_paths = ['/home/oportoles/Documents/MyCode/hipsnp/' +
                       'test_data/datalad/', 
                       '/tmp/genetic/']

    def remove_datalad():
        dl.remove(dataset=pathData, recursive=True, check=False) 

    for pathData in test_data_paths:
        if Path(pathData).is_dir():
            request.addfinalizer(remove_datalad)


def validateJSON(jsonData):
    """attemts to open a JSON var"""
    try:
        json.loads(jsonData)
    except ValueError:
        return False
    return True


# def test_JSON_format_ensembl_human_rsid():
#     """test output is in JSON format"""
#     mock_rsid = 'rs699'
#     outJSON = hps.ensembl_human_rsid(mock_rsid)
#     assert validateJSON(outJSON)


def validateRSTalleles(textOut):
    alleleRef = 'A/G'
    return textOut.find(alleleRef) > 0


@pytest.mark.parametrize("rsid",
                         [('rs699'),
                          ('rs_699'),
                          ('rsid699'),
                          ('rsid_699'),
                          ('RS699'),
                          ('RS_699'),
                          ('699'),
                          ('rs 699')])
def test_ensembl_human_rsid_is_Response(rsid):
    """test output is in JSON format"""
    # mock_rsid = 'rs699'
    outRST = hps.ensembl_human_rsid(rsid)
    assert type(outRST) == Response


@pytest.mark.parametrize("rsid",
                         [('rs699'),
                          ('rs_699'),
                          ('rsid699'),
                          ('rsid_699'),
                          ('RS699'),
                          ('RS_699'),
                          ('699'),
                          ('rs 699')])
def test_ensembl_human_rsid_has_alleles(rsid):
    """test output is in JSON format"""
    # mock_rsid = 'rs699'
    outRST = hps.ensembl_human_rsid(rsid)
    assert validateRSTalleles(outRST.text)


def validatePANDAStype(pdData):
    """Data is a pandas DataFrame with fields 'chromosomes'
     and 'rsids' of type str"""
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
    refColFields = ['rsids', 'chromosomes']
    outPANDAS = hps.rsid2chromosome(mock_rsid)

    valiadtePANDAS_has_RSDIandCROMOSOM(outPANDAS, refColFields)


def test_rsid2chromosome_has_list_of_RSIDandCROMOSOM():
    mock_rsid = ['rs699', 'rs698', 'rs101']
    refColFields = ['rsids', 'chromosomes']
    outPANDAS = hps.rsid2chromosome(mock_rsid)

    valiadtePANDAS_has_RSDIandCROMOSOM(outPANDAS, refColFields)


def removeDATALADdataset(dlObject):
    dlObject.remove()


def test_datalad_get_chromosome_return_DataladType(clean_datalad_Datasets):
    """Read example data from GIN, assert returnst datalad object"""
    c = '1'
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'  # exmaple data
    _, dataladObject, _ = hps.datalad_get_chromosome(c=c, source=source)
    assert type(dataladObject) == dl.Dataset


def validateNameObtainedFiles(dataLget):
    """files obtined with DataLad are the exnple files"""
    filenames = [os.path.basename(ind['path'])
                 for ind in dataLget if ind['type'] == 'file']
    sameFiles = 'example_c1_v0.bgen' and 'example_c1_v0.sample' in filenames
    return sameFiles


def test_datalad_get_chromosome_dataland_get():
    """ dataland.get()_returns_expected_data structure """
    c = '1'
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'  # exmaple data
    with tempfile.TemporaryDirectory() as tempdir:
        _, dataladObject, dataGet = hps.datalad_get_chromosome(c=c, source=source,path=tempdir)
        assert validateNameObtainedFiles(dataGet)


def test_datalad_get_chromosome_file_paths():
    """ returns the path to the files installed by DL """
    c = '1'
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'  # exmaple data

    with tempfile.TemporaryDirectory() as tempdir:
        filesRef = [
            Path(tempdir) / 'genetic' / 'imputation' / 'example_c1_v0.sample',
            Path(tempdir) / 'genetic' / 'imputation' / 'example_c1_v0.bgen' ]
        files, dataladObject, _ = hps.datalad_get_chromosome(c=c,
                                                             source=source,
                                                             path=tempdir)
        assert sorted(files) == sorted(filesRef)


def test_get_chromosome_from_RSIDs():
    """get chromosom from user given RSID, chek return datalad dataset and
    paths to data"""
    datalad_source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'
    rsids = ['rs101']
    chromosomes = ['1']
    # copied from rsid2vcf
    ch_rs = hps.rsid2chromosome(rsids, chromosomes=chromosomes)
    chromosomes = ch_rs['chromosomes'].tolist()
    uchromosomes = pd.unique(chromosomes)
    print('chromosomes needed: ' + str(uchromosomes) + '\n')
    files, ds = [None] * len(uchromosomes), [None] * len(uchromosomes),
    getout = [None] * len(uchromosomes)
    with tempfile.TemporaryDirectory() as tempdir:
        for c in range(len(uchromosomes)):
            ch = uchromosomes[c]
            files[c], ds[c], getout[c] = hps.datalad_get_chromosome(
                ch, source=datalad_source, path=tempdir)
        #
        filesOK = all([type(f) == str or type(f) == PosixPath for f in files[0]])
        dsOK = all([type(d) == dl.Dataset for d in ds])
        assert filesOK and dsOK


# @pytest.mark.parametrize("qctool",
#                         [('/home/oportoles/Apps/qctool_v2.0.6-Ubuntu16.04-x86_64/qctool'),
#                          ('/home/oportoles/Apps/qctool_v2.0.6-Ubuntu16.04-x86_64/'),
#                          ('qctool')])
# def test_rsid2vcf_qctool(qctool):
def test_rsid2vcf(clean_datalad_Datasets):
    """ finds and uses qctool"""
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'
    # outdir = '/home/oportoles/Documents/MyCode/hipsnp/test_data/output'
    # datalad_dir = '/home/oportoles/Documents/MyCode/hipsnp/test_data/datalad'
    # outdir = None # failes test. A path must be given

    rsids = ['RSID_101']
    chromosomes = ['1']
    qctool = '/home/oportoles/Apps/qctool_v2.0.6-Ubuntu16.04-x86_64/qctool'

    # qctool = '/home/oportoles/Apps/qctool_v2.0.6-Ubuntu16.04-x86_64'
    
    # returns: a pandas dataframe with rsid-chromosome pairs
    #  and the vcf files are created in the outdir
    # ch_rs: pandas DataFrame 'chromosomes' 'rsids'
    # files: 
    # ds: datalad DS
    with tempfile.TemporaryDirectory() as tempdir:

        ch_rs, _, dataL = hps.rsid2vcf(rsids,
                                       outdir=tempdir,
                                       datalad_source=source,
                                       qctool=qctool,
                                       datalad_drop=True,
                                       datalad_drop_if_got=True,
                                       datalad_dir=tempdir,
                                       force=False,
                                       chromosomes=chromosomes,
                                       chromosomes_use=None)
        
        assert type(dataL) == dl.Dataset
    # removeDATALADdataset(dataL)

# def test_rsid2vcf_multiple():

#     def rsid2vcf_multiple(rsid_files, outdir,
#                       qctool=None,
#                       datalad_source="ria+http://ukb.ds.inm7.de#~genetic",
#                       datalad_dir=None,
#                       datalad_drop=True):
# 
# inferes the chromosome with a request to ensemble with a rsid


def test_read_vcf():
    path_vcf = ("/home/oportoles/Documents/MyCode/hipsnp/" +
                "test_data/output/chromosome1.vcf")
    pandasDF = hps.read_vcf(path_vcf)
    assert validatePANDAStype(pandasDF)
    # produces an empty dataFrame, all values chormosome1.vcf are column names


def test_vcf2genotype():
    path_vcf = ('/home/oportoles/Documents/MyCode/hipsnp/' +
                'test_data/output/chromosome1.vcf')
    pandasDF = hps.vcf2genotype(path_vcf, th=0.9, snps=None, samples=None)
    assert validatePANDAStype(pandasDF)

# def test_vcf2prs():
# which is weights files?