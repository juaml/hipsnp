import pytest
import hipsnp as hps
import datalad.api as dl
import tempfile
import pandas as pd
from pandas._testing import assert_frame_equal
import numpy as np
import shutil
import copy


def test_read_bgen_for_Genotype_has_metadata():
    """Bgen returned as Genotype object with expected fields and dymensions"""

    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'
    with tempfile.TemporaryDirectory() as tmpdir:
        print(tmpdir + '/')
        dataset = dl.clone(source=source, path=tmpdir + '/')
        dataset.get()
        bgenfile = tmpdir + '/imputation/example_c1_v0.bgen'

        # original code
        snpdata, probsdata = hps.read_bgen(files=bgenfile, 
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


def test_Filter_options():
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
        gen_filt_sample = hps.Filter(gen_ref).by_samples(samples=keep_samples)

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

        gen_filt_rsid = hps.Filter(gen_ref).by_rsids(rsids=keep_rsids)

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

        # linearity of filters
        gen_filt_rsid_from_sample =\
            hps.Filter(gen_filt_sample).by_rsids(rsids=keep_rsids)

        gen_filt_sample_from_rsid =\
            hps.Filter(gen_filt_rsid).by_samples(samples=keep_samples)

        gen_filt_rsid_and_sample = hps.Filter(gen_ref).by_rsids_and_samples(
            rsids=keep_rsids, samples=keep_samples)

        # assert are the same

        assert gen_filt_rsid_from_sample.metadata.equals(
               gen_filt_rsid_and_sample.metadata)
        assert gen_filt_sample_from_rsid.metadata.equals(
               gen_filt_rsid_and_sample.metadata)

        for v1, v2 in zip(gen_filt_rsid_from_sample.probabilities.values(),
                          gen_filt_rsid_and_sample.probabilities.values()):
            assert all(v1[0] == v2[0]) and all(v1[0] == v2[0])

        assert all([all(v1[0] == v2[0]) and all(v1[0] == v2[0]) for v1, v2
                    in zip(gen_filt_rsid_from_sample.probabilities.values(),
                           gen_filt_rsid_and_sample.probabilities.values())])

        assert all([all(v1[0] == v2[0]) and all(v1[0] == v2[0]) for v1, v2
                    in zip(gen_filt_sample_from_rsid.probabilities.values(),
                           gen_filt_rsid_and_sample.probabilities.values())])








