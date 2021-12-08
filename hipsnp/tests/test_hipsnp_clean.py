import copy
import shutil
import tempfile
from pathlib import Path
from _pytest.python_api import raises
import datalad.api as dl
import hipsnp as hps
import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal

# from hipsnp.hipsnp import Genotype
# import hipsnp_origen as hps_o

# This test does not compare the ouputs of the previous verions to the latest.
# This test is intended to be a stand alone test for final package


def validatePANDAScolumns(outPANDAS, refColFields):
    outFields = [field for field in outPANDAS.columns]
    return refColFields.sort() == outFields.sort()


def test_from_bgen_multiple_idnetical_files():
    """The same file is passed multiple times to from_bgen"""
    nFiles = 5
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'
    with tempfile.TemporaryDirectory() as tmpdir:
        print(tmpdir + '/')
        dataset = dl.clone(source=source, path=tmpdir + '/')
        dataset.get()
        bgenfile = [tmpdir + '/imputation/example_c1_v0.bgen']
        bgenfile *= nFiles

        with pytest.raises(ValueError):
            hps.Genotype.from_bgen(files=bgenfile)


def test_from_bgen_files_duplicate_RSID():
    """copy and rename a mock file to have variaous files with same content
    Duplicated RSIDs should be ignored"""
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'
    with tempfile.TemporaryDirectory() as tmpdir:
        print(tmpdir + '/')
        dataset = dl.clone(source=source, path=tmpdir + '/')
        dataset.get()
        bgenfile = tmpdir + '/imputation/example_c1_v0.bgen'
        bgenfile2 = tmpdir + '/imputation/example2_c1_v0.bgen'
        shutil.copy(bgenfile, bgenfile2)

        gen_ref = hps.Genotype.from_bgen(files=bgenfile)
        gen_dup = hps.Genotype.from_bgen(files=[bgenfile, bgenfile2])

        assert_frame_equal(gen_ref.metadata, gen_dup.metadata)
        assert gen_ref.probabilities.keys() == gen_dup.probabilities.keys()
        assert all([np.array_equal(tuple_ref[1], tuple_dup[1], equal_nan=True)
                    for tuple_ref, tuple_dup
                    in zip(gen_ref.probabilities.values(),
                           gen_dup.probabilities.values())
                    ])


@pytest.mark.parametrize("metaCol",
                         [(['REF', 'ALT', 'CHROM']),
                          ([3, 2, 1]),
                          ([None, None]),
                          (['REF', 'ALT', 'CHROM', 'POS', 'ID'])])
def test_Genotype__validate_arguments_column_metadata(metaCol):
    "Force Exception that checks for column names in Genotype metadata"
    
    df = pd.DataFrame(columns=metaCol)
    with pytest.raises(ValueError):
        hps.Genotype(metadata=df, probabilities=None)


def test_Genotype__validate_arguments_rsids():
    "Force Exception that checks same rsids in metadata and probabilites"
  
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'
    with tempfile.TemporaryDirectory() as tmpdir:
        print(tmpdir + '/')
        dataset = dl.clone(source=source, path=tmpdir + '/')
        dataset.get()
        bgenfile = tmpdir + '/imputation/example_c1_v0.bgen'

        gen = hps.Genotype.from_bgen(files=bgenfile)

        gen_modified = copy.deepcopy(gen)
        gen_modified.probabilities.update({'RSID_XX': None})

        with pytest.raises(ValueError):
            hps.Genotype(metadata=gen_modified.metadata,
                         probabilities=gen_modified.probabilities)
        del gen_modified.probabilities['RSID_XX']
        del gen_modified.probabilities['RSID_200']

        with pytest.raises(ValueError):
            hps.Genotype(metadata=gen_modified.metadata,
                         probabilities=gen_modified.probabilities)


def test_Genotype__validate_arguments_probability_dimension():
    "Force Exception that checks the dimension of probabilites"

    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = dl.clone(source=source, path=tmpdir + '/')
        dataset.get()
        bgenfile = tmpdir + '/imputation/example_c1_v0.bgen'

        gen = hps.Genotype.from_bgen(files=bgenfile)

        gen_modified = copy.deepcopy(gen)
        prob   = gen.probabilities['RSID_200'][1]
        sample = gen.probabilities['RSID_200'][0]
        # remove dimension from axis 0
        prob_modified = np.delete(prob, obj=1, axis=0)
        gen_modified.probabilities['RSID_200'] = (sample, prob_modified)

        with pytest.raises(ValueError):
            hps.Genotype(metadata=gen_modified.metadata,
                         probabilities=gen_modified.probabilities)
        # remove dimension from axis 1
        prob_modified = np.delete(prob, obj=1, axis=1)
        gen_modified.probabilities['RSID_200'] = (sample, prob_modified)

        with pytest.raises(ValueError):
            hps.Genotype(metadata=gen_modified.metadata,
                         probabilities=gen_modified.probabilities)


@pytest.mark.parametrize("in_place",
                         [(True),
                          (False)])
def test_filter_options_no_weights(in_place):
    """Test if the filtered out elements are not in the Gentype Object"""
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'
    keep_rsids = ['RSID_2', 'RSID_3', 'RSID_4', 'RSID_5', 'RSID_6',
                  'RSID_7', 'RSID_8', 'RSID_9', 'RSID_10', 'RSID_11']
    n_keep_rsids = len(keep_rsids)
    
    keep_samples = ['sample_001', 'sample_002', 'sample_003', 'sample_004',
                    'sample_005', 'sample_006', 'sample_007', 'sample_008']
    n_keep_samples = len(keep_samples)

    not_a_sample = 'xxxx'

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = dl.clone(source=source, path=tmpdir + '/')
        dataset.get()
        bgenfile = tmpdir + '/imputation/example_c1_v0.bgen'

        gen_ref = hps.Genotype.from_bgen(files=bgenfile)
        n_rsid_mock_data = gen_ref.metadata.index.shape[0]
        n_sample_mock_data =  len(gen_ref.probabilities['RSID_2'][0])

        # Filter by samples, check lentgh of samples and probs, and metadata 
        # content

        if in_place:
            gen_filt_sample = copy.deepcopy(gen_ref)
            gen_filt_sample.filter(samples=keep_samples, inplace=in_place)
        else:
            gen_filt_sample = gen_ref.filter(samples=keep_samples,
                                             inplace=in_place)

        n_filt_samples = np.array([prob[0].shape[0] for prob in
                                   gen_filt_sample.probabilities.values()])
        n_filt_probs = np.array([prob[1].shape[0] for prob in
                                 gen_filt_sample.probabilities.values()])

        assert all(n_filt_samples == n_filt_probs)
        assert all(n_filt_samples == n_keep_samples)
        assert all(n_filt_probs == n_keep_samples)
        assert gen_ref.metadata.equals(gen_filt_sample.metadata)
        # there should be no changes to RSID
        assert len(gen_filt_sample.metadata.index) == n_rsid_mock_data
        assert len(gen_filt_sample.probabilities) == n_rsid_mock_data
        assert gen_filt_sample.metadata.equals(gen_ref.metadata)

        # filter by RSIds
        if in_place:
            gen_filt_rsid = copy.deepcopy(gen_ref)
            gen_filt_rsid.filter(rsids=keep_rsids, inplace=in_place)
        else:
            gen_filt_rsid = gen_ref.filter(rsids=keep_rsids, inplace=in_place)

        n_filt_samples = np.array([prob[0].shape[0] for prob in
                                   gen_filt_rsid.probabilities.values()])
        n_filt_probs = np.array([prob[1].shape[0] for prob in
                                 gen_filt_rsid.probabilities.values()])
    
        # There should be no changes to samples
        assert all(n_filt_samples == n_filt_probs)
        assert all(n_filt_samples == n_sample_mock_data)
        assert all(n_filt_probs == n_sample_mock_data)

        # RSIDs filterd out form metadata and probabilites
        assert n_keep_rsids == gen_filt_rsid.metadata.index.shape[0] 
        assert n_keep_rsids == len(gen_filt_rsid.probabilities.keys())

        assert all(np.isin(gen_filt_rsid.metadata.index, keep_rsids ))
        assert any(np.isin(gen_ref.metadata.index, keep_rsids ))
        assert all([k_rsid in keep_rsids for k_rsid in
                    gen_filt_rsid.probabilities.keys()])

        # filter RSID and Samples
        if in_place:
            gen_filt_rsid_and_sample = copy.deepcopy(gen_ref)
            gen_filt_rsid_and_sample.filter(rsids=keep_rsids,
                                            samples=keep_samples,
                                            inplace=in_place)
        else:
            gen_filt_rsid_and_sample = gen_ref.filter(rsids=keep_rsids,
                                                      samples=keep_samples,
                                                      inplace=in_place)

        n_filt_samples = np.array([prob[0].shape[0] for prob in
            gen_filt_rsid_and_sample.probabilities.values()])
        n_filt_probs = np.array([prob[1].shape[0] for prob in
            gen_filt_rsid_and_sample.probabilities.values()])

        assert all(n_filt_samples == n_filt_probs)
        assert all(n_filt_samples == n_keep_samples)
        assert all(n_filt_probs == n_keep_samples)
        # there are changes to RSID
        assert len(gen_filt_rsid_and_sample.metadata.index) == n_keep_rsids
        assert len(gen_filt_rsid_and_sample.probabilities) == n_keep_rsids

        assert n_keep_rsids == gen_filt_rsid_and_sample.metadata.index.shape[0] 
        assert n_keep_rsids ==\
            len(gen_filt_rsid_and_sample.probabilities.keys())

        assert all(np.isin(gen_filt_rsid_and_sample.metadata.index,
                           keep_rsids))
        assert any(np.isin(gen_ref.metadata.index, keep_rsids ))
        assert all([k_rsid in keep_rsids for k_rsid in
                    gen_filt_rsid_and_sample.probabilities.keys()])
        # no matchn samples to filter specifications
        with pytest.raises(ValueError):
            gen_ref.filter(rsids=None, samples=not_a_sample, inplace=in_place)


@pytest.mark.parametrize("in_place",
                         [(True),
                          (False)])
def test_filter_only_weigths(in_place):
    path_to_weights = '/home/oportoles/Documents/MyCode/hipsnp/test_data/'
    weights_files = ['weights_5.csv', 'weights_100.csv', 'weights_all.csv', 
                     'weights_noMatchRSID.csv', ]
    n_rsid = [5, 100, 199, 0]
    weights = []
    for wfile in weights_files:
        path_to_file = path_to_weights + wfile
        file = hps.read_weights(path_to_file)
        weights.append(file)

    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = dl.clone(source=source, path=tmpdir + '/')
        dataset.get()
        bgenfile = tmpdir + '/imputation/example_c1_v0.bgen'
        gen_ref = hps.Genotype.from_bgen(files=bgenfile)

    for i, w in enumerate(weights):
        gen = copy.deepcopy(gen_ref)
        if n_rsid[i] == 0:
            with pytest.raises(ValueError):
                gen.filter(weights=w)

        else:
            if in_place:
                gen.filter(weights=w, inplace=in_place)
                assert len(gen.rsids) == n_rsid[i]
                assert validatePANDAScolumns(w,
                                             ['ea', 'weight', 'rsid', 'chr'])
            else:
                gen_filt = gen.filter(weights=w, inplace=in_place)
                assert len(gen_filt.rsids) == n_rsid[i]


@pytest.mark.parametrize("in_place",
                         [(True),
                          (False)])
def test_filter_rsids_weights(in_place):

    path_to_weights = '/home/oportoles/Documents/MyCode/hipsnp/test_data/'
    weights_files = ['weights_5.csv', 'weights_5_unsortedRSID.csv',
                     'weights_5_duplicatedRSID.csv']
    
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = dl.clone(source=source, path=tmpdir + '/')
        dataset.get()
        bgenfile = tmpdir + '/imputation/example_c1_v0.bgen'
        gen_ref = hps.Genotype.from_bgen(files=bgenfile)

        # test 1. Less rsids in weights
        rsids_weights_5 = ['RSID_2', 'RSID_3', 'RSID_5', 'RSID_6', 'RSID_7']
        rsids_weights_dup = ['RSID_2', 'RSID_5', 'RSID_6', 'RSID_7']
        keep_rsids = ['RSID_2', 'RSID_3', 'RSID_4', 'RSID_5', 'RSID_6',
                      'RSID_7', 'RSID_8', 'RSID_9', 'RSID_10', 'RSID_11']
        gen = copy.deepcopy(gen_ref)
        for file in weights_files:
            w = path_to_weights + file
            if in_place:
                gen.filter(rsids=keep_rsids, weights=w, inplace=in_place)
                gen_filt = gen
            else:
                gen_filt = gen.filter(rsids=keep_rsids, weights=w,
                                      inplace=in_place)
            if file != 'weights_5_duplicatedRSID.csv':
                assert (sorted(gen_filt.probabilities.keys()) ==
                        sorted(rsids_weights_5))
                assert (sorted(gen_filt.metadata.index) ==
                        sorted(rsids_weights_5))
            else:
                assert (sorted(gen_filt.probabilities.keys()) ==
                        sorted(rsids_weights_dup))
                assert (sorted(gen_filt.metadata.index) ==
                        sorted(rsids_weights_dup))

        # test 2. Less rsids

        gen = copy.deepcopy(gen_ref)
        keep_rsids = ['RSID_2', 'RSID_3']
        for file in weights_files:
            w = path_to_weights + file
            if in_place:
                gen.filter(rsids=keep_rsids, weights=w, inplace=in_place)
                gen_filt = gen
            else:
                gen_filt = gen.filter(rsids=keep_rsids, weights=w,
                                      inplace=in_place)
            if file != 'weights_5_duplicatedRSID.csv':
                assert (sorted(gen_filt.probabilities.keys()) ==
                        sorted(keep_rsids))
                assert (sorted(gen_filt.metadata.index) ==
                        sorted(keep_rsids))
            else:
                assert (sorted(gen_filt.probabilities.keys()) ==
                        sorted(['RSID_2']))
                assert (sorted(gen_filt.metadata.index) ==
                        sorted(['RSID_2']))

        with pytest.raises(ValueError):
            gen.filter(rsids='RSID_2000', weights=w, inplace=in_place)


@pytest.mark.parametrize("in_place",
                         [(True),
                          (False)])
def test_consolidate_Genotype(in_place):
    ''' Samples are reordered in dictionary and probabilites matrix, RIDS are
    removed if not matching smples with other RSIDS, wrong Sample ID. '''
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'
    with tempfile.TemporaryDirectory() as tmpdir:
        print(tmpdir + '/')
        dataset = dl.clone(source=source, path=tmpdir + '/')
        dataset.get()
        bgenfile = tmpdir + '/imputation/example_c1_v0.bgen'
        gen_ref = hps.Genotype.from_bgen(files=bgenfile)
        
    # Test 1 #
    # Randomize samples in one RSID
    # ------ #
    gen_mod = copy.deepcopy(gen_ref)

    rand_idx = np.arange(500)
    np.random.shuffle(rand_idx)
    tmp_tuple_rsid = (gen_ref._probabilities['RSID_3'][0][rand_idx],
                      gen_ref._probabilities['RSID_3'][1][rand_idx, :])
    gen_mod._probabilities['RSID_3'] = tmp_tuple_rsid

    if in_place:
        gen_mod.consolidate(inplace=in_place)
        gen_consol = copy.deepcopy(gen_mod)
        gen_mod = copy.deepcopy(gen_ref)
        gen_mod._probabilities['RSID_3'] = tmp_tuple_rsid 
    else:
        gen_consol = gen_mod.consolidate(inplace=in_place)

    # check that mock data manipulation is effective
    assert any([any(s_ref[0] != s_mod[0]) for s_ref, s_mod in
                zip(gen_ref.probabilities.values(),
                    gen_mod.probabilities.values())])
    
    assert any([np.nansum(p_ref[1] - p_mod[1]) != 0
                for p_ref, p_mod in
                zip(gen_ref.probabilities.values(),
                    gen_mod.probabilities.values())])

    # consolideted data 
    # same metadata
    assert gen_ref.metadata.equals(gen_consol.metadata)
    # sample IDs are rearranged
    assert all([all(s_ref[0] == s_cons[0]) for s_ref, s_cons in
                zip(gen_ref.probabilities.values(),
                    gen_consol.probabilities.values())])
    # probabilites are rearranged
    assert all([np.nansum(p_ref[1] - p_cons[1]) == 0
                for p_ref, p_cons in
                zip(gen_ref.probabilities.values(),
                    gen_consol.probabilities.values())])
   
    # Test 2 #
    # shorten the number of samples in one RSDI and randomize another
    # ------ #
    n_samples = 150
    tmp_tuple_rsid = (gen_ref._probabilities['RSID_20'][0][:n_samples],
                      gen_ref._probabilities['RSID_20'][1][:n_samples, :])
    gen_mod._probabilities['RSID_20'] = tmp_tuple_rsid

    if in_place:
        gen_mod.consolidate(inplace=in_place)
        gen_consol = copy.deepcopy(gen_mod)
        gen_mod = copy.deepcopy(gen_ref)
        gen_mod._probabilities['RSID_20'] = tmp_tuple_rsid 
    else:
        gen_consol = gen_mod.consolidate(inplace=in_place)

    # consolideted data
    # same metadata
    assert gen_ref.metadata.equals(gen_consol.metadata)
    # number of samples is
    # sample IDs are rearranged
    assert all([all(s_ref[0][:n_samples] == s_cons[0]) for s_ref, s_cons in
                zip(gen_ref.probabilities.values(),
                    gen_consol.probabilities.values())])
    # probabilites are rearranged
    assert all([np.nansum(p_ref[1][:n_samples, :] - p_cons[1]) == 0
                for p_ref, p_cons in
                zip(gen_ref.probabilities.values(),
                    gen_consol.probabilities.values())])
    
    # number of samples is n_samples
    assert all([(p_sample[0].shape[0] == n_samples and 
                 p_sample[1].shape[0] == n_samples)
                for p_sample in gen_consol.probabilities.values()])

    # Test 3 #
    # sorten and randomize samples in on RSID
    # -------- #
    gen_mod = copy.deepcopy(gen_ref)
    rand_idx = np.arange(n_samples)
    np.random.shuffle(rand_idx)
    tmp_tuple_rsid = (gen_mod._probabilities['RSID_20'][0][rand_idx],
                      gen_mod._probabilities['RSID_20'][1][rand_idx, :])
    gen_mod._probabilities['RSID_20'] = tmp_tuple_rsid
    
    if in_place:
        gen_mod.consolidate(inplace=in_place)
        gen_consol = copy.deepcopy(gen_mod)
        gen_mod = copy.deepcopy(gen_ref)
        gen_mod._probabilities['RSID_20'] = tmp_tuple_rsid 
    else:
        gen_consol = gen_mod.consolidate(inplace=in_place)

    # consolideted data
    # same metadata
    assert gen_ref.metadata.equals(gen_consol.metadata)
    # number of samples is
    # sample IDs are rearranged
    assert all([all(np.isin(s_ref[0][:n_samples], s_cons[0]))
                for s_ref, s_cons in
                zip(gen_ref.probabilities.values(),
                    gen_consol.probabilities.values())])
    # probabilites are rearranged
    assert all([np.nansum(p_ref[1][:n_samples, :] - p_cons[1]) == 0
                for p_ref, p_cons in
                zip(gen_ref.probabilities.values(),
                    gen_consol.probabilities.values())])
    
    # number of samples is n_samples
    assert all([(p_sample[0].shape[0] == n_samples and
                 p_sample[1].shape[0] == n_samples)
                for p_sample in gen_consol.probabilities.values()])

    # Test 4 #
    # Samples in one RSID cannot be consolidated
    # ------ #
    gen_mod = copy.deepcopy(gen_ref)
    odd_rsid = 'RSID_5'
    # n_rsid = 198
    other_samples = np.array([s[:7] + '10' + s[7:] for s in
                              gen_ref._probabilities[odd_rsid][0]])
    tmp_tuple_rsid = (other_samples,
                      gen_ref._probabilities[odd_rsid][1])
    gen_mod._probabilities[odd_rsid] = tmp_tuple_rsid

    with pytest.raises(ValueError):
        gen_mod.consolidate(inplace=True)


def test__get_array_of_probabilites_and_samples():
    # the matrix of probabilites has the right values afeter consoloidating
    # probabilites
    # generate random probs, create mock Genotype, re order samples, consolidate
    # get probabilites -> they are equal to inital prob.

    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = dl.clone(source=source, path=tmpdir + '/')
        dataset.get()
        bgenfile = tmpdir + '/imputation/example_c1_v0.bgen'
        gen = hps.Genotype.from_bgen(files=bgenfile)

    mockprob = np.random.randn(199, 500, 3)  # num. of rsids & samples source
    for i, rsid_val in enumerate(gen.probabilities.items()):
        gen._probabilities[rsid_val[0]] = (rsid_val[1][0],
                                           np.squeeze(mockprob[i, :, :]))

    with pytest.raises(ValueError):
        gen.get_array_of_probabilities()

    with pytest.raises(ValueError):
        gen.unique_samples()

    gen.consolidate()
    probs = gen.get_array_of_probabilities()
    assert np.array_equal(mockprob, probs)
   
    samples = gen.unique_samples()
    assert isinstance(samples[0], str)
    assert samples.shape[0] == 500


def test_read_weight_wrong_weiths_files():
    path_to_weights = '/home/oportoles/Documents/MyCode/hipsnp/test_data/'
    
    wfile = 'weights_5_duplicatedRSID.csv'
    w = hps.read_weights(path_to_weights + wfile)
    assert w.shape[0] == 4 and w.shape[1] == 3
    assert sorted(list(w.index)) == sorted(['RSID_2', 'RSID_5', 'RSID_6',
                                            'RSID_7'])

    wfile = 'weights_5_other_headers.csv'
    with pytest.raises(ValueError):
        hps.read_weights(path_to_weights + wfile)


def test_rsid_to_bgen_no_qstool():
    qctool = None
    with pytest.raises(ValueError):
        hps.rsid_to_bgen(rsids='rs101', outdir='', qctool=qctool)


def test_rsid_to_bgen_as_before_Genotype():
    """ finds and uses qctool"""
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'
    rsids = ['RSID_101']
    chromosomes = ['1']
    qctool = '/home/oportoles/Apps/qctool_v2.0.6-Ubuntu16.04-x86_64/qctool'

    with tempfile.TemporaryDirectory() as tempdir:

        ch_rs, files, dataL = hps.rsid_to_bgen(rsids,
                                               outdir=tempdir,
                                               datalad_source=source,
                                               qctool=qctool,
                                               datalad_drop=True,
                                               datalad_drop_if_got=True,
                                               data_dir=tempdir,
                                               recompute=False,
                                               chromosomes=chromosomes,
                                               chromosomes_use=None)
        filesRef = [tempdir + '/imputation/' + 'example_c' +
                    str(chromosomes[0]) + '_v0.bgen',
                    tempdir + '/imputation/' + 'example_c' +
                    str(chromosomes[0]) + '_v0.sample']

        assert isinstance(ch_rs, pd.core.frame.DataFrame)
        assert validatePANDAScolumns(ch_rs, ['rsids', 'chromosomes'])
        assert sorted([Path(f) for f in filesRef]) == sorted(files)
        assert type(dataL) == dl.Dataset


def _filesHaveName(dataLget):
    """files obtined with DataLad are the exnple files"""
    filenames = [Path(ind['path']).name
                 for ind in dataLget if ind['type'] == 'file']
    sameFiles = 'example_c1_v0.bgen' and 'example_c1_v0.sample' in filenames
    return sameFiles


def test_get_chromosome_outputTypes_pass():
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'  # exmaple data
    c = '1'
    with tempfile.TemporaryDirectory() as tempdir:
        filesRef = [
            tempdir + '/imputation/' + 'example_c' + str(c) + '_v0.bgen',
            tempdir + '/imputation/' + 'example_c' + str(c) + '_v0.sample']
        files, ds, getout = hps.get_chromosome(c=c, datalad_source=source,
                                               data_dir=tempdir)
        assert sorted([Path(f) for f in filesRef]) == sorted(files)
        assert type(ds) == dl.Dataset
        assert _filesHaveName(getout)

        # chromosome given as int intead of str
        c = 1
        files_i, ds_i, getout_i = hps.get_chromosome(c=c,
                                                     datalad_source=source,
                                                     data_dir=tempdir)
        assert files_i == files
        assert getout_i[0]['message'] == 'already present'

        # no datalad source, files are stored locally
        c = '1'
        files_NO_dl, ds_NO_dl, getout_NO_dl = hps.get_chromosome(
            c=c, datalad_source=None, data_dir=tempdir)
        assert sorted([Path(f) for f in filesRef]) == sorted(files_NO_dl)
        assert ds_NO_dl is None
        assert all([f == 'datalad not used' for f in getout_NO_dl])

        # chormosome 'c' does not match chromosome on datalad sourc
        c = '23'
        with pytest.raises(ValueError):
            hps.get_chromosome(c=c, datalad_source=source, data_dir=tempdir)

    # woring path to chormosome files
    with tempfile.TemporaryDirectory() as tempdir:
        c = '1'
        with pytest.raises(ValueError):
            hps.get_chromosome(c=c, datalad_source=None, data_dir=tempdir)


def test_ensembl_human_rsid_has_alleles():
    """test output is in JSON format"""
    rsidsPass = ['rs699', 'rs102']
    for rsid in rsidsPass:
        outRST = hps.ensembl_human_rsid(rsid)
        assert 'A/G' in outRST['mappings'][0]['allele_string']


def test_ensembl_human_rsid_has_alleles_captures_failsCapital():
    """Exception raised internally"""
    rsidsFail = ['RS699', 'ID699', '699']
    for rsid in rsidsFail:
        with pytest.raises(ValueError):
            hps.ensembl_human_rsid(rsid)


def test_ensembl_human_rsid_has_alleles_captures_give_integer():
    """rsids with wrong format"""
    rsidsFail = [123, 699, 'RS102']
    for rsid in rsidsFail:
        with pytest.raises(ValueError):
            hps.ensembl_human_rsid(rsid)


def test_ensembl_human_rsid_read_RSIDs_csv_and_PGS():
    """rsids given with a csv file"""
    pathfiles = '/home/oportoles/Documents/MyCode/hipsnp/test_data/'
    rsidFile =  pathfiles + 'rsid_699_102.csv'
    rsid = ['rs699', 'rs102']
    # ASK: data test files are stored locally, is there a better way to do it? 
    # put the files on hipsnp/test/data folder and use relative paths
    out_str = hps.rsid2chromosome(rsid)
    out_f = hps.rsid2chromosome(rsidFile)
        
    assert_frame_equal(out_str, out_f)

    pgsfile = pathfiles + 'weights_PGS000001.txt'
    out_pgs = hps.rsid2chromosome(pgsfile)
    
    assert isinstance(out_pgs, pd.core.frame.DataFrame)
    assert out_pgs.shape[0] == 77


def test_rsid_to_genotype_riskscore_mock_Genotype():

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

    # end of moking preparatives

    g_ale, g_012 = mockGen.rsid_to_genotype()
    dosage, risk = mockGen.rsid_to_riskscore(weights=mock_w)
    assert_frame_equal(g_012, mock_g_012)
    assert_frame_equal(g_ale, mock_g_ale)
    assert_frame_equal(dosage, mock_dosage)
    assert_frame_equal(risk, mock_risk)
