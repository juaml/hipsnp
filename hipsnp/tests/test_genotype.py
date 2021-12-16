import tempfile
import numpy as np
import pandas as pd
import datalad.api as dl

from pandas._testing import assert_frame_equal
import pytest

import hipsnp as hps
from hipsnp.hipsnp import read_weights


@pytest.mark.parametrize(
    "columns,isgood", [
        (['REF', 'ALT', 'CHROM'], False),
        ([3, 2, 1], False),
        ([None, None], False),
        (['REF', 'ALT', 'CHROM', 'POS', 'ID'], False),
        (['REF', 'ALT', 'CHROM', 'POS', 'ID', 'FORMAT'], True),
    ])
def test_validate_arguments_column_metadata(columns, isgood):
    "Force Exception that checks for column names in Genotype metadata"

    df = pd.DataFrame(columns=columns)
    if isgood:
        hps.Genotype(metadata=df, probabilities={})
    else:
        with pytest.raises(ValueError, match='columns in metadata'):
            hps.Genotype(metadata=df, probabilities=None)


def test_validate_arguments_rsids():
    "Force Exception that checks same rsids in metadata and probabilities"

    source = 'https://gin.g-node.org/juaml/datalad-example-bgen'
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = dl.install(source=source, path=tmpdir + '/')  # type: ignore
        dataset.get()
        bgenfile = tmpdir + '/imputation/example_c1_v0.bgen'

        gen = hps.read_bgen(files=bgenfile)

    gen_modified = gen._clone()
    gen_modified.probabilities.update({'RSID_XX': None})

    with pytest.raises(ValueError, match='Mismatch of RSIDs'):
        hps.Genotype(metadata=gen_modified.metadata,
                     probabilities=gen_modified.probabilities)
    del gen_modified.probabilities['RSID_XX']
    del gen_modified.probabilities['RSID_200']

    with pytest.raises(ValueError, match='Mismatch of RSIDs'):
        hps.Genotype(metadata=gen_modified.metadata,
                     probabilities=gen_modified.probabilities)

    gen_modified = gen._clone()
    prob   = gen.probabilities['RSID_200'][1]
    sample = gen.probabilities['RSID_200'][0]
    # remove dimension from axis 0
    prob_modified = np.delete(prob, obj=1, axis=0)
    gen_modified.probabilities['RSID_200'] = (sample, prob_modified)
    with pytest.raises(ValueError, match='Dimension mismatch'):
        hps.Genotype(metadata=gen_modified.metadata,
                     probabilities=gen_modified.probabilities)

    # remove dimension from axis 1
    prob_modified = np.delete(prob, obj=1, axis=1)
    gen_modified.probabilities['RSID_200'] = (sample, prob_modified)
    with pytest.raises(ValueError, match='Dimension mismatch'):
        hps.Genotype(metadata=gen_modified.metadata,
                     probabilities=gen_modified.probabilities)


def test_filter():
    """Test if the filtered out elements are not in the Gentype Object"""
    source = 'https://gin.g-node.org/juaml/datalad-example-bgen'
    keep_rsids = ['RSID_2', 'RSID_3', 'RSID_4', 'RSID_5', 'RSID_6',
                  'RSID_7', 'RSID_8', 'RSID_9', 'RSID_10', 'RSID_11']
    n_keep_rsids = len(keep_rsids)

    keep_samples = ['sample_001', 'sample_002', 'sample_003', 'sample_004',
                    'sample_005', 'sample_006', 'sample_007', 'sample_008']
    n_keep_samples = len(keep_samples)

    not_a_sample = 'xxxx'

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = dl.install(source=source, path=tmpdir + '/')  # type: ignore
        dataset.get()
        bgenfile = tmpdir + '/imputation/example_c1_v0.bgen'

        gen_ref = hps.read_bgen(files=bgenfile)

    n_rsid_mock_data = gen_ref.metadata.index.shape[0]
    n_sample_mock_data =  len(gen_ref.probabilities['RSID_2'][0])

    # Inplace filter by samples
    gen_filt_sample = gen_ref._clone()
    gen_filt_sample.filter(samples=keep_samples, inplace=True)

    n_filt_samples = np.array(
        [prob[0].shape[0] for prob in
            gen_filt_sample.probabilities.values()])
    n_filt_probs = np.array(
        [prob[1].shape[0] for prob in
            gen_filt_sample.probabilities.values()])

    assert all(n_filt_samples == n_filt_probs)
    assert all(n_filt_samples == n_keep_samples)
    assert all(n_filt_probs == n_keep_samples)
    assert gen_ref.metadata.equals(gen_filt_sample.metadata)
    # there should be no changes to RSID
    assert len(gen_filt_sample.metadata.index) == n_rsid_mock_data
    assert len(gen_filt_sample.probabilities) == n_rsid_mock_data
    assert gen_filt_sample.metadata.equals(gen_ref.metadata)

    # Not inplace
    gen_filt_sample2 = gen_ref.filter(samples=keep_samples, inplace=False)
    hps.utils.testing.assert_genotype_equal(
        gen_filt_sample, gen_filt_sample2)

    # Inplace filter by rsids
    gen_filt_rsid = gen_ref._clone()
    gen_filt_rsid.filter(rsids=keep_rsids, inplace=True)

    n_filt_samples = np.array(
        [prob[0].shape[0] for prob in gen_filt_rsid.probabilities.values()])
    n_filt_probs = np.array(
        [prob[1].shape[0] for prob in gen_filt_rsid.probabilities.values()])

    # There should be no changes to samples
    assert all(n_filt_samples == n_filt_probs)
    assert all(n_filt_samples == n_sample_mock_data)
    assert all(n_filt_probs == n_sample_mock_data)

    # RSIDs filterd out form metadata and probabilities
    assert n_keep_rsids == gen_filt_rsid.metadata.index.shape[0]
    assert n_keep_rsids == len(gen_filt_rsid.probabilities.keys())

    assert all(np.isin(gen_filt_rsid.metadata.index, keep_rsids ))
    assert any(np.isin(gen_ref.metadata.index, keep_rsids ))
    assert all([k_rsid in keep_rsids for k_rsid in
                gen_filt_rsid.probabilities.keys()])

    # Not inplace
    gen_filt_rsid2 = gen_ref.filter(rsids=keep_rsids, inplace=False)
    hps.utils.testing.assert_genotype_equal(gen_filt_rsid, gen_filt_rsid2)

    # Inplace filter by RSID and samples
    gen_filt_both = gen_ref._clone()
    gen_filt_both.filter(
        rsids=keep_rsids, samples=keep_samples, inplace=True)

    n_filt_samples = np.array(
        [prob[0].shape[0] for prob in gen_filt_both.probabilities.values()])
    n_filt_probs = np.array(
        [prob[1].shape[0] for prob in gen_filt_both.probabilities.values()])

    assert all(n_filt_samples == n_filt_probs)
    assert all(n_filt_samples == n_keep_samples)
    assert all(n_filt_probs == n_keep_samples)
    # there are changes to RSID
    assert len(gen_filt_both.metadata.index) == n_keep_rsids
    assert len(gen_filt_both.probabilities) == n_keep_rsids

    assert n_keep_rsids == gen_filt_both.metadata.index.shape[0]
    assert n_keep_rsids ==\
        len(gen_filt_both.probabilities.keys())

    assert all(np.isin(gen_filt_both.metadata.index, keep_rsids))
    assert any(np.isin(gen_ref.metadata.index, keep_rsids))
    assert all([k_rsid in keep_rsids for k_rsid in
                gen_filt_both.probabilities.keys()])

    # Not inplace
    gen_filt_both2 = gen_ref.filter(rsids=keep_rsids, samples=keep_samples,
                                    inplace=False)
    hps.utils.testing.assert_genotype_equal(gen_filt_both, gen_filt_both2)

    # no matching samples to filter specifications
    with pytest.raises(ValueError, match='samples matching filter'):
        gen_ref.filter(rsids=None, samples=not_a_sample, inplace=False)

    with pytest.warns(RuntimeWarning, match='Nothing to filter'):
        gen_ref.filter(rsids=None, samples=None, inplace=False)

    # Use weights
    data_path = hps.utils.testing.get_testing_data_dir()

    weights_files = {
        'weights_5.csv': 5,
        'weights_100.csv': 100,
        'weights_all.csv': 199,
    }

    for weights_file, n_rsid in weights_files.items():
        t_w = read_weights(data_path / weights_file)
        t_g = gen_ref.filter(
            weights=data_path / weights_file, inplace=False)
        t_g2 = gen_ref.filter(weights=t_w, inplace=False)
        t_g3 = gen_ref.filter(
            weights=(data_path / weights_file).as_posix(), inplace=False)
        hps.utils.testing.assert_genotype_equal(t_g, t_g2)
        hps.utils.testing.assert_genotype_equal(t_g, t_g3)
        assert len(t_g.rsids) == n_rsid

    with pytest.raises(ValueError, match='No RSIDs matching'):
        gen_ref.filter(weights=data_path / 'weights_noMatchRSID.csv')

    rsids_weights_5 = ['RSID_2', 'RSID_3', 'RSID_5', 'RSID_6', 'RSID_7']
    rsids_weights_dup = ['RSID_2', 'RSID_5', 'RSID_6', 'RSID_7']
    keep_rsids = ['RSID_2', 'RSID_3', 'RSID_4', 'RSID_5', 'RSID_6',
                  'RSID_7', 'RSID_8', 'RSID_9', 'RSID_10', 'RSID_11']

    for fname in ['weights_5.csv', 'weights_5_unsortedRSID.csv']:
        gen_filt = gen_ref.filter(
            rsids=keep_rsids, weights=data_path / fname,
            inplace=False)
        assert (sorted(gen_filt.probabilities.keys()) ==
                sorted(rsids_weights_5))
        assert (sorted(gen_filt.metadata.index) ==
                sorted(rsids_weights_5))

    fname = 'weights_5_duplicatedRSID.csv'
    with pytest.warns(RuntimeWarning, match='duplicated RSIDs'):
        gen_filt = gen_ref.filter(
            rsids=keep_rsids, weights=data_path / fname,
            inplace=False)
    assert (sorted(gen_filt.probabilities.keys()) ==
            sorted(rsids_weights_dup))
    assert (sorted(gen_filt.metadata.index) ==
            sorted(rsids_weights_dup))

    # Less RSIDS than weights
    keep_rsids = ['RSID_2', 'RSID_3']
    for fname in [ 'weights_5.csv', 'weights_5_unsortedRSID.csv']:
        gen_filt = gen_ref.filter(
            rsids=keep_rsids, weights=data_path / fname,
            inplace=False)
        assert (sorted(gen_filt.probabilities.keys()) ==
                sorted(keep_rsids))
        assert (sorted(gen_filt.metadata.index) ==
                sorted(keep_rsids))

    fname = 'weights_5_duplicatedRSID.csv'
    with pytest.warns(RuntimeWarning, match='duplicated RSIDs'):
        gen_filt = gen_ref.filter(
            rsids=keep_rsids, weights=data_path / fname,
            inplace=False)
    assert (sorted(gen_filt.probabilities.keys()) ==
            sorted(['RSID_2']))
    assert (sorted(gen_filt.metadata.index) ==
            sorted(['RSID_2']))

    fname = 'weights_5.csv'
    with pytest.raises(ValueError, match='No RSIDs matching'):
        gen_ref.filter(
            rsids='RSID_2000', weights=data_path / fname, inplace=False)


def test_consolidate():
    """Test consolidating a genotype"""
    source = 'https://gin.g-node.org/juaml/datalad-example-bgen'

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = dl.install(source=source, path=tmpdir + '/')  # type: ignore
        dataset.get()
        bgenfile = tmpdir + '/imputation/example_c1_v0.bgen'

        gen_ref = hps.read_bgen(files=bgenfile)

    # 1. Randomize samples in one RSID
    gen_mod = gen_ref._clone()
    np.random.seed(10)
    rand_idx = np.arange(500)
    np.random.shuffle(rand_idx)
    tmp_tuple_rsid = (gen_ref._probabilities['RSID_3'][0][rand_idx],
                      gen_ref._probabilities['RSID_3'][1][rand_idx, :])
    gen_mod._probabilities['RSID_3'] = tmp_tuple_rsid

    # check that mock data manipulation is effective
    assert any([any(s_ref[0] != s_mod[0]) for s_ref, s_mod in
                zip(gen_ref.probabilities.values(),
                    gen_mod.probabilities.values())])

    assert any([np.nansum(p_ref[1] - p_mod[1]) != 0
                for p_ref, p_mod in
                zip(gen_ref.probabilities.values(),
                    gen_mod.probabilities.values())])

    assert not gen_mod.is_consolidated
    gen_consol = gen_mod.consolidate(inplace=False)
    assert gen_consol.is_consolidated

    # Check consolidated metadata (same)
    assert gen_ref.metadata.equals(gen_consol.metadata)

    # sample IDs are rearranged
    for ((s_ref, p_ref), (s_cons, p_cons)) in zip(
            gen_ref.probabilities.values(),
            gen_consol.probabilities.values()):
        np.testing.assert_array_equal(s_ref, s_cons)
        np.testing.assert_array_equal(p_ref, p_cons)

    gen_consol2 = gen_mod._clone()
    gen_consol2.consolidate(inplace=True)

    hps.utils.testing.assert_genotype_equal(gen_consol, gen_consol2)

    # 2. Shorten the number of samples in one RSID and randomize another
    n_samples = 150
    tmp_tuple_rsid = (gen_ref._probabilities['RSID_20'][0][:n_samples],
                      gen_ref._probabilities['RSID_20'][1][:n_samples, :])
    gen_mod._probabilities['RSID_20'] = tmp_tuple_rsid
    gen_consol = gen_mod.consolidate(inplace=False)

    # Check consolidated metadata (same)
    assert gen_ref.metadata.equals(gen_consol.metadata)

    for ((s_ref, p_ref), (s_cons, p_cons)) in zip(
            gen_ref.probabilities.values(), gen_consol.probabilities.values()):
        np.testing.assert_array_equal(s_ref[:n_samples], s_cons)
        np.testing.assert_array_equal(p_ref[:n_samples], p_cons)

    gen_consol2 = gen_mod._clone()
    gen_consol2.consolidate(inplace=True)

    hps.utils.testing.assert_genotype_equal(gen_consol, gen_consol2)

    # 3. Shorten and randomize samples in one RSID
    gen_mod = gen_ref._clone()
    rand_idx = np.arange(n_samples)
    np.random.shuffle(rand_idx)
    tmp_tuple_rsid = (gen_mod._probabilities['RSID_20'][0][rand_idx],
                      gen_mod._probabilities['RSID_20'][1][rand_idx, :])
    gen_mod._probabilities['RSID_20'] = tmp_tuple_rsid

    gen_consol = gen_mod.consolidate(inplace=False)
    # Check consolidated metadata (same)
    assert gen_ref.metadata.equals(gen_consol.metadata)

    for ((s_ref, p_ref), (s_cons, p_cons)) in zip(
            gen_ref.probabilities.values(), gen_consol.probabilities.values()):
        np.testing.assert_array_equal(s_ref[:n_samples], s_cons)
        np.testing.assert_array_equal(p_ref[:n_samples], p_cons)

    gen_consol2 = gen_mod._clone()
    gen_consol2.consolidate(inplace=True)

    hps.utils.testing.assert_genotype_equal(gen_consol, gen_consol2)

    # 4. Samples in one RSID cannot be consolidated
    gen_mod = gen_ref._clone()
    odd_rsid = 'RSID_5'
    other_samples = np.array([s[:7] + '10' + s[7:] for s in
                              gen_ref._probabilities[odd_rsid][0]])
    tmp_tuple_rsid = (other_samples,
                      gen_ref._probabilities[odd_rsid][1])
    gen_mod._probabilities[odd_rsid] = tmp_tuple_rsid

    with pytest.raises(ValueError, match='There are no common samples'):
        gen_mod.consolidate(inplace=True)


def test__consolidated_probabilities():
    """Test obtaining the consolidated probabilities"""
    source = 'https://gin.g-node.org/juaml/datalad-example-bgen'

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = dl.install(source=source, path=tmpdir + '/')  # type: ignore
        dataset.get()
        bgenfile = tmpdir + '/imputation/example_c1_v0.bgen'

        gen_ref = hps.read_bgen(files=bgenfile)

    with pytest.raises(ValueError, match='are not consolidated'):
        gen_ref._consolidated_probabilities()

        with pytest.raises(ValueError, match='are not consolidated'):
            gen_ref._consolidated_samples()

    np.random.seed(10)
    mockprob = np.random.randn(199, 500, 3)  # num. of rsids & samples source
    for i, rsid_val in enumerate(gen_ref.probabilities.items()):
        gen_ref._probabilities[rsid_val[0]] = \
            (rsid_val[1][0], np.squeeze(mockprob[i, :, :]))

    gen_ref.consolidate()
    probs = gen_ref._consolidated_probabilities()
    assert np.array_equal(mockprob, probs)

    samples = gen_ref._consolidated_samples()
    assert isinstance(samples[0], str)
    assert samples.shape[0] == 500


def test_alleles_riskscore():
    """Test the alleles and riskscore method of the Genotype class"""
    source = 'https://gin.g-node.org/juaml/datalad-example-bgen'

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = dl.install(source=source, path=tmpdir + '/')  # type: ignore
        dataset.get()
        bgenfile = tmpdir + '/imputation/example_c1_v0.bgen'
        gen = hps.read_bgen(files=bgenfile)

    mock_meta = pd.DataFrame(data=np.array([['A', 'G', 'x', 'x', 'x', 'x'],
                                            ['G', 'A', 'x', 'x', 'x', 'x']]),
                             index=['RSID_2', 'RSID_3'],
                             columns=['REF', 'ALT', 'CHROM',
                                      'POS', 'ID', 'FORMAT'])
    mock_samples = gen.probabilities['RSID_3'][0][:3]
    mock_prob = {
        'RSID_2': (mock_samples, np.array([[0.25, 0.25, 0.50],
                                           [0.50, 0.25, 0.25],
                                           [0.25, 0.50, 0.25]])),
        'RSID_3': (mock_samples, np.array([[1.0, 0.0, 0.0],
                                           [0.0, 1.0, 0.0],
                                           [0.0, 0.0, 1.0]]))}

    mockGen = hps.Genotype(mock_meta, mock_prob)

    mock_g_012 = pd.DataFrame(data=np.array([[2, 0, 1], [0, 1, 2]]),
                              index=['RSID_2', 'RSID_3'],
                              columns=mock_samples)

    mock_g_ale = pd.DataFrame(data=np.array([['GG', 'AA', 'AG'],
                                             ['GG', 'AG', 'AA']]),
                              index=['RSID_2', 'RSID_3'],
                              columns=mock_samples)

    mock_dosage = pd.DataFrame(data=np.array([[0.75, 1.25, 1.0],    # weight=1
                                              [0.00, 1.00, 2.0]]),  # weight=2
                               index=['RSID_2', 'RSID_3'],
                               columns=mock_samples)

    mock_risk = pd.DataFrame(data=np.array([0.75, 3.25, 5.0]),
                             index=mock_samples)

    data_path = hps.utils.testing.get_testing_data_dir()
    mock_w = data_path / 'weights_5.csv'

    g_ale, g_012 = mockGen.alleles()
    dosage, risk = mockGen.riskscores(weights=mock_w)
    assert_frame_equal(g_012, mock_g_012)
    assert_frame_equal(g_ale, mock_g_ale)
    assert_frame_equal(dosage, mock_dosage)
    assert_frame_equal(risk, mock_risk)

    # test 2: filter by rsid
    mock_risk_filt_rs = np.array([[0.75], [1.25], [1.0]])

    g_ale, g_012 = mockGen.alleles(rsids='RSID_2')
    dosage, risk = mockGen.riskscores(rsids='RSID_2', weights=mock_w)
    np.testing.assert_array_equal(
        g_012.loc['RSID_2'].values, mock_g_012.loc['RSID_2'].values)
    np.testing.assert_array_equal(
        g_ale.loc['RSID_2'].values, mock_g_ale.loc['RSID_2'].values)
    np.testing.assert_array_equal(
        dosage.loc['RSID_2'].values, mock_dosage.loc['RSID_2'].values)
    np.testing.assert_array_equal(mock_risk_filt_rs, risk.values)
