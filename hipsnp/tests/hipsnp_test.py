import pytest
import os
from requests.models import Response
import hipsnp as hps
import json
import pandas as pd
import datalad.api as dl
import tempfile

def filesHaveName(dataLget):
    """files obtined with DataLad are the exnple files"""
    filenames = [os.path.basename(ind['path'])
                 for ind in dataLget if ind['type'] == 'file']
    sameFiles = 'example_c1_v0.bgen' and 'example_c1_v0.sample' in filenames
    return sameFiles

@pytest.mark.parametrize("c",[('1'),('23'),(None),(1)])
def test_get_chromosome_outputTypes(c):
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'  # exmaple data
    
    with tempfile.TemporaryDirectory() as tempdir:
        filesRef = [tempdir + '/imputation/' + 'example_c' + str(c) + '_v0.bgen',
                    tempdir + '/imputation/' + 'example_c' + str(c) + '_v0.sample']
        errors = []

        files, ds, getout = hps.get_chromosome(c=c, datalad_source=source, data_dir=tempdir)
        if not sorted(filesRef) == sorted(files):
            errors.append('Error: wrong data paths')
        if not type(ds) == dl.Dataset:
            errors.append('Error: not a Datalad Dataset')
        if not filesHaveName(getout):
            errors.append('Error: wrong data files')
        assert not errors, "errors occured:\n{}".format("\n".join(errors))

def test_get_chromosome_output_Datalad():
    c = '1'
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'  # exmaple data
    
    with tempfile.TemporaryDirectory() as tempdir: 
        filesRef = [tempdir + '/imputation/' + 'example_c' + str(c) + '_v0.bgen',
                    tempdir + '/imputation/' + 'example_c' + str(c) + '_v0.sample']
        files, ds, getout = hps.get_chromosome(c=c, datalad_source=source, data_dir=tempdir)
        assert sorted(filesRef) == sorted(files)
        assert type(ds) == dl.Dataset
        assert filesHaveName(getout)


def test_get_chromosome_output_None():
    c = '1'
    source = None  # exmaple data
    
    with tempfile.TemporaryDirectory() as tempdir: 
        _, ds, getout  = hps.get_chromosome(c=c, datalad_source=source, data_dir=tempdir)
        assert ds == None
        assert len(getout) == 2



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


def validateRSTalleles(textOut):
    alleleRef = 'A/G'
    return textOut.find(alleleRef) > 0


def test_ensembl_human_rsid_has_alleles():
    """test output is in JSON format"""
    rsidsPass = ['rs699', 'rs102']
    for rsid in  rsidsPass:
        outRST = hps.ensembl_human_rsid(rsid)
        assert 'A/G' in outRST.text


def test_ensembl_human_rsid_has_alleles_captures_failsCapital():
    """Exception raised internally"""
    rsidsFail = ['RS699', 'ID699', '699']
    for rsid in  rsidsFail:    
        with pytest.raises(ValueError):
            hps.ensembl_human_rsid(rsid)


def test_ensembl_human_rsid_has_alleles_captures_give_integer():
    """Exception raised internally"""
    rsidsFail = [123, 699]
    for rsid in  rsidsFail:    
        with pytest.raises(TypeError):
            hps.ensembl_human_rsid(rsid)


def validatePANDAScolumns(outPANDAS, refColFields):
    outFields = [field for field in outPANDAS.columns]
    return refColFields.sort() == outFields.sort()


@pytest.mark.parametrize("rsid,chromosome",
                         [('rs699', None),
                          ('rs699', '1'),
                          ('rs699', 'a'),
                          (['rs699'], [None]),
                          (['rs699'], ['1']),
                          (['rs699'], ['a']),
                          (['rs699','rs694'],['1','2']),
                          (['rs699','rs694'],['1',None]),
                          (['rs699','rs694'],['1','1']),
                          (['rs699','rs699'],['1','1'])])
def test_rsid2chromosome_has_pandas_format(rsid,chromosome):
    outPANDAS = hps.rsid2chromosome(rsid, chromosomes=chromosome)
    errors = []
    if not isinstance(outPANDAS, pd.core.frame.DataFrame):
        errors.append('Error: is not a Panads Dataframe')
    if not validatePANDAScolumns(outPANDAS,['rsids', 'chromosomes']):
        errors.append('Error: wrong PANDAS columns')
    assert not errors, "errors occured:\n{}".format("\n".join(errors))


def test_rsid2snp():
    """ finds and uses qctool"""
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'
    rsids = ['RSID_101']
    chromosomes = ['1']
    qctool = '/home/oportoles/Apps/qctool_v2.0.6-Ubuntu16.04-x86_64/qctool'

    with tempfile.TemporaryDirectory() as tempdir:

        ch_rs, files, dataL = hps.rsid2snp(rsids,
                                           outdir=tempdir,
                                           datalad_source=source,
                                           qctool=qctool,
                                           datalad_drop=True,
                                           datalad_drop_if_got=True,
                                           data_dir=tempdir,
                                           force=False,
                                           chromosomes=chromosomes,
                                           chromosomes_use=None,
                                           outformat='bgen')
        filesRef = [tempdir + '/imputation/' + 'example_c' + str(chromosomes[0]) + '_v0.bgen',
                    tempdir + '/imputation/' + 'example_c' + str(chromosomes[0]) + '_v0.sample']
  
        errors = []
        if not isinstance(ch_rs, pd.core.frame.DataFrame):
            errors.append('Error: is not a Panads Dataframe')
        if not validatePANDAScolumns(ch_rs, ['rsids', 'chromosomes']):
            errors.append('Error: wrong PANDAS columns')
        if not sorted(filesRef) == sorted(files):
            errors.append('Error: wrong data paths')
        if not type(dataL) == dl.Dataset:
            errors.append('Error: not a Datalad Dataset')
        assert not errors, "errors occured:\n{}".format("\n".join(errors))


def test_rsid2snp_multiple():

    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'
    rsids = ['rs101']
    chromosomes = ['1']
    qctool = '/home/oportoles/Apps/qctool_v2.0.6-Ubuntu16.04-x86_64/qctool'
    
    with tempfile.TemporaryDirectory() as tempdir:        
        ch_rs, files, dataL = hps.rsid2snp(rsids,
                                           outdir=tempdir,
                                           datalad_source=source,
                                           qctool=qctool,
                                           datalad_drop=False, # is True, there is not data for next call
                                           datalad_drop_if_got=False,
                                           data_dir=tempdir,
                                           force=False,
                                           chromosomes=chromosomes,
                                           chromosomes_use=None,
                                           outformat='bgen')

        outdirs = hps.rsid2snp_multiple(files=[tempdir + '/rsids_chromosome1.txt'], 
                                        outdir=tempdir,
                                        qctool=qctool,
                                        datalad_source=source,
                                        data_dir=tempdir,
                                        datalad_drop=True,
                                        outformat='bgen')
        
        filesRef = tempdir +'/rsids_chromosome1'
  
        assert filesRef == outdirs


def test_read_bgen():
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'
    rsids = ['RSID_101']
    chromosomes = ['1']
    qctool = '/home/oportoles/Apps/qctool_v2.0.6-Ubuntu16.04-x86_64/qctool'

    with tempfile.TemporaryDirectory() as tempdir:        
        ch_rs, files, dataL = hps.rsid2snp(rsids,
                                           outdir=tempdir,
                                           datalad_source=source,
                                           qctool=qctool,
                                           datalad_drop=False,
                                           # if False, read_bgen cannot find them
                                           datalad_drop_if_got=True,
                                           data_dir=tempdir,
                                           force=False,
                                           chromosomes=chromosomes,
                                           chromosomes_use=None,
                                           outformat='bgen')
        bgenFiles = [tempdir + '/imputation/example_c1_v0.bgen']
        snpdata, probsdata = hps.read_bgen(files=bgenFiles, 
                                            rsids_as_index=True, 
                                            no_neg_samples=False, 
                                            join='inner', 
                                            verify_integrity=False, 
                                            probs_in_pd=False,
                                            verbose=True)
        errors = []
        if not isinstance(probsdata, dict):
            errors.append('Error: wrong data type. It is not a Dict')
        if not isinstance(snpdata, pd.core.frame.DataFrame):
            errors.append('Error: wrong data type. It is not a Pandas DataFrame')
        assert not errors, "errors occured:\n{}".format("\n".join(errors))


def test_read_weights_mock():
    """Mock weights file pass asserts"""

    pathMock = '/home/oportoles/Documents/MyCode/hipsnp/test_data/weights.csv'
    w = hps.read_weights(pathMock)
    assert  validatePANDAScolumns(w, ['ea', 'weight', 'rsid', 'chr'])


def test_read_vcf():
    """ outputs pandas dataframe with cols"""
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'
    rsids = ['RSID_101']
    chromosomes = ['1']
    qctool = '/home/oportoles/Apps/qctool_v2.0.6-Ubuntu16.04-x86_64/qctool'

    with tempfile.TemporaryDirectory() as tempdir:        
        ch_rs, files, dataL = hps.rsid2snp(rsids,
                                           outdir=tempdir,
                                           datalad_source=source,
                                           qctool=qctool,
                                           datalad_drop=False,
                                           # if False, read_bgen cannot find them
                                           datalad_drop_if_got=True,
                                           data_dir=tempdir,
                                           force=False,
                                           chromosomes=chromosomes,
                                           chromosomes_use=None,
                                           outformat='vcf')

        bgenFiles = [tempdir + '/chromosome1.vcf']
        snpdata, _ = hps.read_bgen(files=bgenFiles, 
                                    rsids_as_index=True, 
                                    no_neg_samples=False, 
                                    join='inner', 
                                    verify_integrity=False, 
                                    probs_in_pd=False,
                                    verbose=True)
       
        assert validatePANDAScolumns(snpdata, ['CHROM', 'POS', 'ID', 'REF', 'ALT','QUAL','FILTER', 'INFO'])

