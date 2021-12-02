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
# import hipsnp_origen as hps_o
from . import hipsnp_origen as hps_o


def test_read_bgen_for_Genotype_has_metadata():
    """Bgen returned as Genotype object with expected fields and dymensions"""

    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'
    with tempfile.TemporaryDirectory() as tmpdir:
        print(tmpdir + '/')
        dataset = dl.clone(source=source, path=tmpdir + '/')
        dataset.get()
        bgenfile = tmpdir + '/imputation/example_c1_v0.bgen'

        # original code
        snpdata, probsdata = hps_o.read_bgen(files=bgenfile,
                                             rsids_as_index=True,
                                             no_neg_samples=False,
                                             join='inner',
                                             verify_integrity=False,
                                             probs_in_pd=False,
                                             verbose=True)

        gen = hps.Genotype.from_bgen(files=[bgenfile])

        assert_frame_equal(snpdata, gen.metadata)

        # probabilites sum to one
        assert all([np.nanmean(np.sum(gen.probabilities[k_key][1], axis=1))
                    == 1 for k_key in gen.probabilities.keys()])


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
def test_filter_options(in_place):
    """Test if the filtered out elements are not in the Gentype Object"""
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'
    keep_rsids = ['RSID_2', 'RSID_3', 'RSID_4', 'RSID_5', 'RSID_6',
                  'RSID_7', 'RSID_8', 'RSID_9', 'RSID_10', 'RSID_11']
    n_keep_rsids = len(keep_rsids)
    
    keep_samples = ['sample_001', 'sample_002', 'sample_003', 'sample_004',
                    'sample_005', 'sample_006', 'sample_007', 'sample_008']
    n_keep_samples = len(keep_samples)

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


def test__get_array_of_probabilites():
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

    gen.consolidate()
    probs = gen.get_array_of_probabilities()
    assert np.array_equal(mockprob, probs)


def test_filter_by_weigths():
    path_to_weights = '/home/oportoles/Documents/MyCode/hipsnp/test_data/'
    weights_files = ['weights_5.csv', 'weights_100.csv', 'weights_all.csv', 
                     'weights_noMatchRSID.csv']
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
                gen.filter_by_weigths(w)

        else:
            gen.filter_by_weigths(w)
            assert len(gen.rsids) == n_rsid[i]
            assert validatePANDAScolumns(w, ['ea', 'weight', 'rsid', 'chr'])


def test_snp2genotype():
    "Compare outputs of snp2genotyp in Genotype object ans as initial function"

    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = dl.clone(source=source, path=tmpdir + '/')
        dataset.get()
        bgenfile = tmpdir + '/imputation/example_c1_v0.bgen'
        gen = hps.Genotype.from_bgen(files=bgenfile)

        snpdata, probsdata = hps_o.read_bgen(files=bgenfile,
                                             rsids_as_index=True,
                                             no_neg_samples=False,
                                             join='inner',
                                             verify_integrity=False,
                                             probs_in_pd=False,
                                             verbose=True)
 
    g_allele_old, g_012_old, prob_old =\
        hps_o.snp2genotype(snpdata=snpdata,
                           th=0.9,
                           snps=None,
                           samples=None,
                           genotype_format='allele',
                           probs=probsdata,
                           weights=None,
                           verbose=True,
                           profiler=None)

    gen.consolidate()
    g_allele, g_012 = gen.snp2genotype()

    # assert np.array_equal(g_012_old.to_numpy().T, g_012)
    # assert np.array_equal(g_allele_old.to_numpy().T, g_allele)
    assert_frame_equal(g_012_old.transpose(), g_012, check_dtype=False)
    assert_frame_equal(g_allele_old.transpose(), g_allele)
    # SOME esceptions about the format need to be added


def test_read_weights():
    path_to_weights = '/home/oportoles/Documents/MyCode/hipsnp/test_data/'
    weights_files = ['weights_5.csv', 'weights_100.csv', 'weights_all.csv'] 
 
    for wf in weights_files:
        path_w = path_to_weights + wf
        weights = hps.read_weights(path_w)
        weights_old = hps_o.read_weights(path_w)

        assert np.all(weights.to_numpy() == weights_old.to_numpy()[:, 1:])


def test_snp2genotype_weigths():
    """Compare outputs of snp2genotyp in Genotype object ans as initial function
    when weights are given"""

    # path_weights = '/home/oportoles/Documents/MyCode/hipsnp/test_data/weights_all.csv'
    
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = dl.clone(source=source, path=tmpdir + '/')
        dataset.get()
        bgenfile = tmpdir + '/imputation/example_c1_v0.bgen'
        gen = hps.Genotype.from_bgen(files=bgenfile)

        snpdata, probsdata = hps_o.read_bgen(files=bgenfile,
                                             rsids_as_index=True,
                                             no_neg_samples=False,
                                             join='inner',
                                             verify_integrity=False,
                                             probs_in_pd=False,
                                             verbose=True)
 
    gen.consolidate()
    gen_ref = copy.deepcopy(gen)
    # weights = hps.read_weights_Genotype(path_weights)
    # weights_old = hps.read_weights(path_to_weights)
    path_to_weights = '/home/oportoles/Documents/MyCode/hipsnp/test_data/'
    weights_files = ['weights_5.csv', 'weights_100.csv', 'weights_all.csv']

    for wf in weights_files:
        path_w = path_to_weights + wf
        g_allele_old, g_012_old, risk_old =\
            hps_o.snp2genotype(snpdata=snpdata,
                               th=0.9,
                               snps=None,
                               samples=None,
                               genotype_format='allele',
                               probs=probsdata,
                               weights=path_w,
                               verbose=True,
                               profiler=None)
        gen = copy.deepcopy(gen_ref)
        g_allele, g_012, risk = gen.snp2genotype(weights=path_w)

        # assert np.array_equal(g_012_old.to_numpy().T, g_012)
        # assert np.array_equal(g_allele_old.to_numpy().T, g_allele)
        
        # assert np.allclose(np.squeeze(risk_old.to_numpy()), risk,
        #                    equal_nan=True)
        assert_frame_equal(risk_old, risk)


def test_rsid2snp():
    """ original and new function give the same outputs"""
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'
    rsids = ['RSID_101']
    chromosomes = ['1']
    qctool = '/home/oportoles/Apps/qctool_v2.0.6-Ubuntu16.04-x86_64/qctool'

    with tempfile.TemporaryDirectory() as tempdir:

        ch_rs_old, files_old, dataL_old =\
            hps_o.rsid2snp(rsids,
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
                                           chromosomes_use=None)

    assert ch_rs_old.equals(ch_rs)


def test_rsid2snp_no_qstool():
    qctool = None
    with pytest.raises(ValueError):
        hps.rsid2snp(rsids='rs101', outdir='', qctool=qctool)


def test_rsid2snp_noDrop():
    """ original and new function give the same outputs when datalad datasets
    are not dropped"""
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'
    rsids = ['RSID_101']
    chromosomes = ['1']
    qctool = '/home/oportoles/Apps/qctool_v2.0.6-Ubuntu16.04-x86_64/qctool'

    with tempfile.TemporaryDirectory() as tempdir:

        ch_rs_old, files_old, dataL_old =\
            hps_o.rsid2snp(rsids,
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
                                           chromosomes_use=None)

    assert ch_rs_old.equals(ch_rs)


def validatePANDAScolumns(outPANDAS, refColFields):
    outFields = [field for field in outPANDAS.columns]
    return refColFields.sort() == outFields.sort()


def test_rsid2snp_as_before_Genotype():
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
                                           chromosomes_use=None)
        filesRef = [tempdir + '/imputation/' + 'example_c' +
                    str(chromosomes[0]) + '_v0.bgen',
                    tempdir + '/imputation/' + 'example_c' +
                    str(chromosomes[0]) + '_v0.sample']

        assert isinstance(ch_rs, pd.core.frame.DataFrame)
        assert validatePANDAScolumns(ch_rs, ['rsids', 'chromosomes'])
        assert sorted([Path(f) for f in filesRef]) == sorted(files)
        assert type(dataL) == dl.Dataset


def test_get_chromosome():
    """compare old and new function's outputs
    """
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'  # exmaple data
    cs = '1'
    with tempfile.TemporaryDirectory() as tempdir:
        files, ds, getout = hps.get_chromosome(c=cs,
                                               datalad_source=source,
                                               data_dir=tempdir)
        files_o, ds_o, getout_o = hps_o.get_chromosome(c=cs,
                                                       datalad_source=source,
                                                       data_dir=tempdir)
        assert files == files_o
        assert ds == ds_o


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

        # no datalad source, files are stored locally
        files_NO_dl, ds_NO_dl, getout_NO_dl = hps.get_chromosome(
            c=c, datalad_source=None, data_dir=tempdir)
        assert sorted([Path(f) for f in filesRef]) == sorted(files_NO_dl)
        assert ds_NO_dl is None
        assert all([f == 'datalad not used' for f in getout_NO_dl])

        # chromosome given as int intead of str
        c = 1
        with pytest.raises(TypeError):
            hps.get_chromosome(c=c, datalad_source=source, data_dir=tempdir)

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
    out_str = hps.rsid2chromosome(rsid)
    out_f = hps.rsid2chromosome(rsidFile)
        
    assert_frame_equal(out_str, out_f)

    pgsfile = pathfiles + 'weights_PGS000001.txt'
    out_pgs = hps.rsid2chromosome(pgsfile)
    
    assert isinstance(out_pgs, pd.core.frame.DataFrame)
    assert out_pgs.shape[0] == 77
