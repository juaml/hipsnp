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

from hipsnp.hipsnp import Genotype

def filesHaveName(dataLget):
    """files obtined with DataLad are the exnple files"""
    filenames = [os.path.basename(ind['path'])
                 for ind in dataLget if ind['type'] == 'file']
    sameFiles = 'example_c1_v0.bgen' and 'example_c1_v0.sample' in filenames
    return sameFiles

def test_get_chromosome_outputTypes_pass():
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'  # exmaple data
    cs = ['1', 1]
    with tempfile.TemporaryDirectory() as tempdir:

        errors = []
        for c in cs:
            filesRef = [tempdir + '/imputation/' + 'example_c' + str(c) + '_v0.bgen',
            tempdir + '/imputation/' + 'example_c' + str(c) + '_v0.sample']
            files, ds, getout = hps.get_chromosome(c=c, datalad_source=source, data_dir=tempdir)
            if not sorted(filesRef) == sorted(files):
                errors.append('Error: wrong data paths')
            if not type(ds) == dl.Dataset:
                errors.append('Error: not a Datalad Dataset')
            if not filesHaveName(getout):
                errors.append('Error: wrong data files')
            assert not errors, "errors occured:\n{}".format("\n".join(errors))


def test_get_chromosome_outputTypes_failes():
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'  # exmaple data

    cs = ['23', None]
    with tempfile.TemporaryDirectory() as tempdir:
        for c in cs:
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
            assert len(errors) > 0
            print( "errors occured:\n{}".format("\n".join(errors)))


def test_get_chromosome_NO_datalad_source():
    c = '1'
    source = None  # exmaple data
    
    with tempfile.TemporaryDirectory() as tempdir: 
        with pytest.raises(UnboundLocalError):
            hps.get_chromosome(c=c, datalad_source=source, data_dir=tempdir)
            # ds is not defined on the first if satement


# def test_get_chromosome_output_None():
#     c = '1'
#     source = None  # exmaple data
    
#     with tempfile.TemporaryDirectory() as tempdir: 
#         _, ds, getout  = hps.get_chromosome(c=c, datalad_source=source, data_dir=tempdir)
#         assert ds == None
#         assert len(getout) == 2



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


# @pytest.mark.parametrize("rsid,chromosome",
#                          [('rs699', None),
#                           ('rs699', '1'),
#                           ('rs699', 'a'),
#                           (['rs699'], [None]),
#                           (['rs699'], ['1']),
#                           (['rs699'], ['a']),
#                           (['rs699','rs694'],['1','2']),
#                           (['rs699','rs694'],['1',None]),
#                           (['rs699','rs694'],['1','1']),
#                           (['rs699','rs699'],['1','1'])])
# def test_rsid2chromosome_has_pandas_format(rsid,chromosome):
#     outPANDAS = hps.rsid2chromosome(rsid, chromosomes=chromosome)
#     errors = []
#     if not isinstance(outPANDAS, pd.core.frame.DataFrame):
#         errors.append('Error: is not a Panads Dataframe')
#     if not validatePANDAScolumns(outPANDAS,['rsids', 'chromosomes']):
#         errors.append('Error: wrong PANDAS columns')
#     assert not errors, "errors occured:\n{}".format("\n".join(errors))


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
  
        assert filesRef == outdirs[0]


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
        # bgenFiles = [tempdir + '/chromosome_1.bgen']
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
        hps.GP2dosage(mock_GP, mock_REF, mock_ALT, mock_EA) # snp is not defined in print statement

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
            # docstring says that it read from snp file paths, but it does not have it implemented
            hps.snp2genotype(snpdata = files,
                            th=0.9,
                            snps=None, 
                            samples=None, 
                            genotype_format='allele',
                            probs=None, 
                            weights=None, 
                            verbose=True, 
                            profiler=None)


def pandas_has_ANY_NaN(data):
    if isinstance(data, pd.DataFrame):
        return data.isnull().values.any()
    elif isinstance(data, pd.Series):
        return any([np.isnan(data_i).any() for data_i in data])
    else:
        raise AttributeError

def pandas_has_ALL_NaN(data):
    if isinstance(data, pd.DataFrame):
        return data.isnull().values.all()
    elif isinstance(data, pd.Series):
        return all([np.isnan(data_i).all() for data_i in data])
    else:
        raise AttributeError

def test_snp2genotype_from_BGENfile():
    """ Compute SNP from mock data in bgen format, read_bgen, then get genotype
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
        
        # when probs are not given, the aoutput are empty        
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
       # snpsdata has <9 columns,
        # so line 445 in hispnp returns an empty list

        # with probs given
        geno_allele, geno_012, prob = hps.snp2genotype(snpdata = snpdata,
                                                        th=0.9,
                                                        snps=None, 
                                                        samples=None, 
                                                        genotype_format='allele',
                                                        probs=probsdata,
                                                        weights=None, 
                                                        verbose=True, 
                                                        profiler=None)
        
        assert not geno_allele.empty
        assert not geno_012.empty
        assert not prob.empty
        assert not pandas_has_ANY_NaN(geno_allele)
        assert not pandas_has_ANY_NaN(geno_012) 

        assert pandas_has_ALL_NaN(prob)

        # NaNs because todo fix in line 542 - 546? probability are only assigned to nan values

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

        assert not pandas_has_ANY_NaN(geno_allele)
        assert not pandas_has_ANY_NaN(geno_012)
        assert pandas_has_ANY_NaN(prob) 

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

        assert not pandas_has_ANY_NaN(geno_allele)
        assert not pandas_has_ANY_NaN(geno_012)
        assert not pandas_has_ANY_NaN(risk) # NaNs because todo fix in line 542?
        assert risk.sum().values[0] == 0 # weights are set to None becasue no matching snp and rsid
        # EA  = weights['ea'][weights['rsid'] == snp].values

        # with weights with RSID_101
        weights_mock = '/home/oportoles/Documents/MyCode/hipsnp/test_data/weights_withRSID_101.csv'
        geno_allele, geno_012, risk = hps.snp2genotype(snpdata = snpdata,
                                        th=0.9,
                                        snps=None, 
                                        samples=None, 
                                        genotype_format='allele',
                                        probs=probsdata, 
                                        weights=weights_mock, 
                                        verbose=True, 
                                        profiler=None)

        assert not pandas_has_ANY_NaN(geno_allele)
        assert not pandas_has_ANY_NaN(geno_012)
        assert not pandas_has_ANY_NaN(risk) # NaNs because todo fix in line 542?
        assert risk.sum().values[0] > 0 # weights are set to None becasue no matching snp and rsid
        # EA  = weights['ea'][weights['rsid'] == snp].values

def test_snp2genotype_from_BGENfile_with_repeated_rsids():
    """ Compute SNP from mock data in bgen format, read_bgen, then get genotype
        Assert that output values are not nan"""
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'
    rsids = ['RSID_101', 'RSID_101',]
    chromosomes = ['1', '1']
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

        # with probs given
        geno_allele, geno_012, prob = hps.snp2genotype(snpdata = snpdata,
                                                        th=0.9,
                                                        snps=None, 
                                                        samples=None, 
                                                        genotype_format='allele',
                                                        probs=probsdata,
                                                        weights=None, 
                                                        verbose=True, 
                                                        profiler=None)
        
        assert not geno_allele.empty
        assert not geno_012.empty
        assert not prob.empty
        assert not pandas_has_ANY_NaN(geno_allele)
        assert not pandas_has_ANY_NaN(geno_012) 

        assert pandas_has_ALL_NaN(prob)


def test_read_from_bgen():
    """Function read_began and Genotype.read_from_bgen have the same output"""
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
        
        # make use of datalad.get to obtain the data.


        bgenFiles = [tempdir + '/imputation/example_c1_v0.bgen']
        # bgenFiles = [tempdir + '/chromosome_1.bgen']
        snpdata, probsdata = hps.read_bgen(files=bgenFiles, 
                                            rsids_as_index=True, 
                                            no_neg_samples=False, 
                                            join='inner', 
                                            verify_integrity=False, 
                                            probs_in_pd=False,
                                            verbose=True)

        Genotype.read_from_bgen(files=bgenFiles, 
                                rsids_as_index=True, 
                                no_neg_samples=False, 
                                join='inner', 
                                verify_integrity=False, 
                                probs_in_pd=False,
                                verbose=True)

        assert_frame_equal(snpdata, Genotype.bgenDF)
        assert np.array_equal(np.squeeze(probsdata['0']['probs']), Genotype.sample_probs[rsids[0]][1])
        assert np.array_equal(probsdata['0']['samples'], Genotype.sample_probs[rsids[0]][0])


def test_read_from_bgen_multiple_files():
    """Function read_began and Genotype.read_from_bgen have the same output"""
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'
    rsids = ['RSID_101']
    chromosomes = ['1']
    qctool = '/home/oportoles/Apps/qctool_v2.0.6-Ubuntu16.04-x86_64/qctool'
    nFiles = 5

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
        bgenFiles *= nFiles
        snpdata, probsdata = hps.read_bgen(files=bgenFiles, 
                                            rsids_as_index=True, 
                                            no_neg_samples=False, 
                                            join='inner', 
                                            verify_integrity=False, 
                                            probs_in_pd=False,
                                            verbose=True)

        Genotype.read_from_bgen(files=bgenFiles, 
                                rsids_as_index=True, 
                                no_neg_samples=False, 
                                join='inner', 
                                verify_integrity=False, 
                                probs_in_pd=False,
                                verbose=True)

        assert_frame_equal(snpdata, Genotype.bgenDF)
        for i in range(nFiles):
            assert np.array_equal(np.squeeze(probsdata[str(i)]['probs']), Genotype.sample_probs[rsids[0]][1])
            assert np.array_equal(probsdata[str(i)]['samples'], Genotype.sample_probs[rsids[0]][0])

