import pytest
import os
from requests.models import Response
import hipsnp as hps
import json
import pandas as pd
import numpy as np
from pandas._testing import assert_frame_equal
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
        # bgenFiles = [tempdir + '/imputation/example_c1_v0.bgen']
        bgenFiles = [tempdir + '/chromosome_1.bgen']
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

        bgenFiles = [tempdir + '/chromosome_1.vcf']
        snpdata, _ = hps.read_vcf(files=bgenFiles, 
                                  format=['GP', 'GT:GP'],
                                  no_neg_samples=False, \
                                  join='inner',
                                  verify_integrity=False,
                                  verbose=True)
       
        assert validatePANDAScolumns(snpdata, ['CHROM', 'POS', 'ID', 'REF', 'ALT','QUAL','FILTER', 'INFO'])

def test_read_vcf_bgen_equal():
    """Both files give the same output"""
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'
    rsids = ['RSID_101']
    chromosomes = ['1']
    qctool = '/home/oportoles/Apps/qctool_v2.0.6-Ubuntu16.04-x86_64/qctool'

    with tempfile.TemporaryDirectory() as tempdir:        
        # VCF
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

        bgenFiles = [tempdir + '/chromosome_1.vcf']
        snpdataVCF, _ = hps.read_vcf(files=bgenFiles, 
                                  format=['GP', 'GT:GP'],
                                  no_neg_samples=False, \
                                  join='inner',
                                  verify_integrity=False,
                                  verbose=True)
        # BGEN
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

        bgenFiles = [tempdir + '/chromosome_1.bgen']
        snpdataBGEN, probsdata = hps.read_bgen(files=bgenFiles, 
                                            rsids_as_index=True, 
                                            no_neg_samples=False, 
                                            join='inner', 
                                            verify_integrity=False, 
                                            probs_in_pd=False,
                                            verbose=True)

        with pytest.raises(AssertionError):
            assert_frame_equal(snpdataBGEN, snpdataVCF ) # fails, 

        for column in snpdataBGEN.columns:
            if column == 'POS' or column == 'ID':
                with pytest.raises(AssertionError):
                    assert snpdataBGEN[column].equals(snpdataVCF[column])
            else:
                assert snpdataBGEN[column].equals(snpdataVCF[column])




def test_GP2dosage_operations():
    """Same solution for both compuations"""
    mock_GP  = pd.DataFrame(np.ones((3, 3)))
    mock_REF = ['b', 'a']
    mock_ALT = ['a', 'b']
    mock_EA  = 'a'
    dosis = []
    for i in range(2):
        dosis.append(hps.GP2dosage(mock_GP, mock_REF[i], mock_ALT[i], mock_EA))
    assert all(dosis[0] == dosis[1])
    # assertin based on teh equations in the code, if the code is wrong the test as well.

def test_GP2dosage_missmatch_EA_RE_ALT():
    mock_GP  = pd.DataFrame(np.ones((3, 3)))
    mock_REF = ['b']
    mock_ALT = ['b']
    mock_EA  = 'a'
    with pytest.raises(NameError):
        hps.GP2dosage(mock_GP, mock_REF, mock_ALT, mock_EA) # snp is not defined

def test_snp2genotype_from_SNPfiles():
    """compute SNP from mock data, then get genotype"""
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'
    rsids = ['RSID_101']
    chromosomes = ['1']
    qctool = '/home/oportoles/Apps/qctool_v2.0.6-Ubuntu16.04-x86_64/qctool'

    with tempfile.TemporaryDirectory() as tempdir:

        _, files, _ = hps.rsid2snp(rsids,
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
        
        with pytest.raises(AttributeError): # not implemted reading from snp files
            hps.snp2genotype(snpdata = files,
                            th=0.9,
                            snps=None, 
                            samples=None, 
                            genotype_format='allele',
                            probs=None, 
                            weights=None, 
                            verbose=True, 
                            profiler=None)


def pandas_has_NaN(data):
    if isinstance(data, pd.DataFrame):
        return data.isnull().values.any()
    elif isinstance(data, pd.Series):
        return any([np.isnan(data_i).any() for data_i in data])
    else:
        raise AttributeError


def test_snp2genotype_from_VCFfile():
    """ Compute SNP from mock data in vcf format, read_vcf, then get genotype
        Assert that output values are not nan"""
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
                                           datalad_drop_if_got=True,
                                           data_dir=tempdir,
                                           force=False,
                                           chromosomes=chromosomes,
                                           chromosomes_use=None,
                                           outformat='vcf')

        bgenFiles = [tempdir + '/chromosome_1.vcf']
        snpdata, _ = hps.read_vcf(files=bgenFiles, 
                                  format=['GP', 'GT:GP'],
                                  no_neg_samples=False, \
                                  join='inner',
                                  verify_integrity=False,
                                  verbose=True)
        
        geno_allele, geno_012, prob = hps.snp2genotype(snpdata = snpdata,
                                                        th=0.9,
                                                        snps=None, 
                                                        samples=None, 
                                                        genotype_format='allele',
                                                        probs=None, 
                                                        weights=None, 
                                                        verbose=True, 
                                                        profiler=None)
        
        assert not pandas_has_NaN(geno_allele)
        assert not pandas_has_NaN(geno_012)
        assert pandas_has_NaN(prob) # is it expected to be NaNs?


def test_snp2genotype_from_BGENfile():
    """ Compute SNP from mock data in vcf format, read_vcf, then get genotype
        Assert that output values are not nan"""
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
                                           datalad_drop_if_got=True,
                                           data_dir=tempdir,
                                           force=False,
                                           chromosomes=chromosomes,
                                           chromosomes_use=None,
                                           outformat='bgen')
        # bgenFiles = [tempdir + '/imputation/example_c1_v0.bgen']
        bgenFiles = [tempdir + '/chromosome_1.bgen']
        snpdata, probsdata = hps.read_bgen(files=bgenFiles, 
                                            rsids_as_index=True, 
                                            no_neg_samples=False, 
                                            join='inner', 
                                            verify_integrity=False, 
                                            probs_in_pd=False,
                                            verbose=True)
        
        geno_allele, geno_012, prob = hps.snp2genotype(snpdata = snpdata,
                                                        th=0.9,
                                                        snps=None, 
                                                        samples=None, 
                                                        genotype_format='allele',
                                                        probs=None, 
                                                        weights=None, 
                                                        verbose=True, 
                                                        profiler=None)
        
        assert geno_allele.empty
        assert geno_012.empty
        assert prob.empty 
        # is it expected that they are empty? snpsdata has <9 columns,
        # so line 445 in hispnp returns an empty list

        # With probabilities given by read_bgen
        geno_allele, geno_012, prob = hps.snp2genotype(snpdata = snpdata,
                                                th=0.9,
                                                snps=None, 
                                                samples=None, 
                                                genotype_format='allele',
                                                probs=probsdata, 
                                                weights=None, 
                                                verbose=True, 
                                                profiler=None)

        assert not pandas_has_NaN(geno_allele)
        assert not pandas_has_NaN(geno_012)
        assert pandas_has_NaN(prob) # NaNs because todo fix in line 542?

        # with weights
        weights_mock = '/home/oportoles/Documents/MyCode/hipsnp/test_data/weights.csv'
        geno_allele, geno_012, risk = hps.snp2genotype(snpdata = snpdata,
                                        th=0.9,
                                        snps=None, 
                                        samples=None, 
                                        genotype_format='allele',
                                        probs=probsdata, 
                                        weights=weights_mock, 
                                        verbose=True, 
                                        profiler=None)

        assert not pandas_has_NaN(geno_allele)
        assert not pandas_has_NaN(geno_012)
        assert not pandas_has_NaN(risk) # NaNs because todo fix in line 542?