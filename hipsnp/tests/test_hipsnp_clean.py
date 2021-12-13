import copy
import shutil
import tempfile
from pathlib import Path
import datalad.api as dl
import hipsnp as hps
import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal


# This test does not compare the ouputs of the previous verions to the latest.
# This test is intended to be a stand alone test for final package

def validatePANDAScolumns(outPANDAS, refColFields):
    outFields = [field for field in outPANDAS.columns]
    return refColFields.sort() == outFields.sort()

def test_read_weight_wrong_weiths_files():
    path_to_weights = \
        str(Path().cwd().joinpath("hipsnp", "tests", "test_data"))
    wfile = '/weights_5_duplicatedRSID.csv'
    w = hps.read_weights(path_to_weights + wfile)
    assert w.shape[0] == 4 and w.shape[1] == 3
    assert sorted(list(w.index)) == sorted(['RSID_2', 'RSID_5', 'RSID_6',
                                            'RSID_7'])

    wfile = '/weights_5_other_headers.csv'
    with pytest.raises(ValueError):
        hps.read_weights(path_to_weights + wfile)


def test_pruned_bgen_from_Datalad_no_qstool():
    qctool = None
    with pytest.raises(ValueError):
        hps.pruned_bgen_from_Datalad(rsids='rs101', outdir='', qctool=qctool)


def test_pruned_bgen_from_Datalad_wrong_rsdis_chromosome_size():
     qctool = '/home/oportoles/Apps/qctool_v2.0.6-Ubuntu16.04-x86_64/qctool'
     rsids = ['RSID_2', 'RSID_3', 'RSID_5', 'RSID_6', 'RSID_7']
     chromosomes = ['1']
     with pytest.raises(ValueError):
         hps.pruned_bgen_from_Datalad(rsids=rsids,
                                      chromosomes=chromosomes,
                                      outdir='',
                                      qctool=qctool)


 def test_pruned_bgen_from_Datalad_give_rsdis_chromosome():
     source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'
     qctool = '/home/oportoles/Apps/qctool_v2.0.6-Ubuntu16.04-x86_64/qctool'
     rsids = ['RSID_2', 'RSID_3', 'RSID_5', 'RSID_6', 'RSID_7']
     chromosomes = ['1'] * len(rsids)
     with tempfile.TemporaryDirectory() as tempdir:

         ch_rs, files, dataL = hps.pruned_bgen_from_Datalad(
             rsids,
             outdir=tempdir,
             datalad_source=source,
             qctool=qctool,
             datalad_drop=True,
             datalad_drop_if_got=True,
             data_dir=tempdir,
             recompute=False,
             chromosomes=chromosomes)
     assert sorted(ch_rs['rsids']) == sorted(rsids)
     assert len(files) == 2


def test_pruned_bgen_from_Datalad_as_before_Genotype():
    """ finds and uses qctool"""
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'
    rsids = ['RSID_101']
    chromosomes = ['1']
    qctool = '/home/oportoles/Apps/qctool_v2.0.6-Ubuntu16.04-x86_64/qctool'
    with tempfile.TemporaryDirectory() as tempdir:

        ch_rs, files, dataL = hps.pruned_bgen_from_Datalad(
            rsids,
            outdir=tempdir,
            datalad_source=source,
            qctool=qctool,
            datalad_drop=True,
            datalad_drop_if_got=True,
            data_dir=tempdir,
            recompute=False,
            chromosomes=chromosomes)
        filesRef = [tempdir + '/imputation/' + 'example_c' +
                    str(chromosomes[0]) + '_v0.bgen',
                    tempdir + '/imputation/' + 'example_c' +
                    str(chromosomes[0]) + '_v0.sample']

        assert isinstance(ch_rs, pd.core.frame.DataFrame)
        assert validatePANDAScolumns(ch_rs, ['rsids', 'chromosomes'])
        assert sorted([Path(f) for f in filesRef]) == sorted(files)
        assert type(dataL) == dl.Dataset
        assert not(any(Path(f).is_file() for f in files))
        new_bgen_file = tempdir + '/chromosome' + chromosomes[0] + '.bgen'
        assert Path(new_bgen_file).is_file()

def _filesHaveName(dataLget):
    """files obtined with DataLad are the exnple files"""
    filenames = [Path(ind['path']).name
                 for ind in dataLget if ind['type'] == 'file']
    sameFiles = 'example_c1_v0.bgen' and 'example_c1_v0.sample' in filenames
    return sameFiles


def test_get_chromosome_data_outputTypes_pass():
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'  # exmaple data
    c = '1'
    with tempfile.TemporaryDirectory() as tempdir:
        filesRef = [
            tempdir + '/imputation/' + 'example_c' + str(c) + '_v0.bgen',
            tempdir + '/imputation/' + 'example_c' + str(c) + '_v0.sample']
        files, ds, getout = hps.get_chromosome_data(c=c, datalad_source=source,
                                                    data_dir=tempdir)
        assert sorted([Path(f) for f in filesRef]) == sorted(files)
        assert type(ds) == dl.Dataset
        assert _filesHaveName(getout)

        # chromosome given as int intead of str
        c = 1
        files_i, _, getout_i = hps.get_chromosome_data(c=c,
                                                       datalad_source=source,
                                                       data_dir=tempdir)
        assert files_i == files
        assert getout_i[0]['message'] == 'already present'

        # no datalad source, files are stored locally
        c = '1'
        files_NO_dl, ds_NO_dl, getout_NO_dl = hps.get_chromosome_data(
            c=c, datalad_source=None, data_dir=tempdir)
        assert sorted([Path(f) for f in filesRef]) == sorted(files_NO_dl)
        assert ds_NO_dl is None
        assert all([f == 'datalad not used' for f in getout_NO_dl])

        # chormosome 'c' does not match chromosome on datalad sourc
        c = '23'
        with pytest.raises(ValueError):
            hps.get_chromosome_data(c=c,
                                    datalad_source=source,
                                    data_dir=tempdir)

    # woring path to chormosome files
    with tempfile.TemporaryDirectory() as tempdir:
        c = '1'
        with pytest.raises(ValueError):
            hps.get_chromosome_data(c=c, datalad_source=None, data_dir=tempdir)


def test_request_ensembl_rsid_has_alleles():
    """test output is in JSON format"""
    rsidsPass = ['rs699', 'rs102']
    for rsid in rsidsPass:
        outRST = hps.request_ensembl_rsid(rsid)
        assert 'A/G' in outRST['mappings'][0]['allele_string']


def test_request_ensembl_rsid_has_alleles_captures_failsCapital():
    """Exception raised internally"""
    rsidsFail = ['RS699', 'ID699', '699']
    for rsid in rsidsFail:
        with pytest.raises(ValueError):
            hps.request_ensembl_rsid(rsid)


def test_request_ensembl_rsid_has_alleles_captures_give_integer():
    """rsids with wrong format"""
    rsidsFail = [123, 699, 'RS102']
    for rsid in rsidsFail:
        with pytest.raises(ValueError):
            hps.request_ensembl_rsid(rsid)


def test_request_ensembl_rsid_read_RSIDs_csv_and_PGS():
    """rsids given with a csv file"""
    pathfiles = str(Path().cwd().joinpath("hipsnp", "tests", "test_data"))
    rsidFile =  pathfiles + '/rsid_699_102.csv'
    rsid = ['rs699', 'rs102']

    out_str = hps.rsid_chromosome_DataFrame(rsid)
    out_f = hps.rsid_chromosome_DataFrame(rsidFile)
        
    assert_frame_equal(out_str, out_f)

    pgsfile = pathfiles + '/weights_PGS000001.txt'
    out_pgs = hps.rsid_chromosome_DataFrame(pgsfile)
    
    assert isinstance(out_pgs, pd.core.frame.DataFrame)
    assert out_pgs.shape[0] == 77


def test_alleles_riskscore_mock_Genotype():

    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'  # exmaple data

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = dl.clone(source=source, path=tmpdir + '/')
        dataset.get()
        bgenfile = tmpdir + '/imputation/example_c1_v0.bgen'
        gen = hps.Genotype.from_bgen(files=bgenfile)

    mock_meta = gen.metadata.loc[['RSID_2', 'RSID_3']]
    mock_samples = gen.probabilities['RSID_2'][0][:2]
    mock_prob = {'RSID_2': (mock_samples, np.array([[0.25, 0.25, 0.5],
                                                    [0.5, 0.25, 0.25]])),
                 'RSID_3': (mock_samples, np.array([[1.0, 0.0, 0.0],
                                                    [0.0, 1.0, 0.0]]))}

    mockGen = hps.Genotype(mock_meta, mock_prob)

    mock_g_012 = pd.DataFrame(data=np.array([[2, 0], [0, 1]]),
                              index=['RSID_2', 'RSID_3'],
                              columns=mock_samples)

    mock_g_ale = pd.DataFrame(data=np.array([['GG', 'AA'], ['AA', 'AG']]),
                              index=['RSID_2', 'RSID_3'],
                              columns=mock_samples)

    mock_dosage = pd.DataFrame(data=np.array([
                                             [0.75, 1.25],
                                             [2.0, 1.0]
                                             ]),
                               index=['RSID_2', 'RSID_3'],
                               columns=mock_samples)

    mock_risk = pd.DataFrame(data=np.array([4.75, 3.25]),
                             index=mock_samples)

    mock_w = '/home/oportoles/Documents/MyCode/hipsnp/test_data/weights_5.csv'
    mock_w = str(Path().cwd().joinpath("hipsnp",
                                       "tests",
                                       "test_data",
                                       "weights_5.csv"))

    # end of moking preparatives

    g_ale, g_012 = mockGen.alleles()
    dosage, risk = mockGen.riskscore(weights=mock_w)
    assert_frame_equal(g_012, mock_g_012)
    assert_frame_equal(g_ale, mock_g_ale)
    assert_frame_equal(dosage, mock_dosage)
    assert_frame_equal(risk, mock_risk)

    # test 2: filter by rsid
    mock_risk_filt_rs = np.array([[0.75], [1.25]])

    g_ale, g_012 = mockGen.alleles(rsids='RSID_2')
    dosage, risk = mockGen.riskscore(rsids='RSID_2', weights=mock_w)
    assert np.array_equal(g_012.loc['RSID_2'].values,
                          mock_g_012.loc['RSID_2'].values)
    assert np.array_equal(g_ale.loc['RSID_2'].values,
                          mock_g_ale.loc['RSID_2'].values)
    assert np.array_equal(dosage.loc['RSID_2'].values,
                          mock_dosage.loc['RSID_2'].values)
    assert np.array_equal(mock_risk_filt_rs, risk.values)