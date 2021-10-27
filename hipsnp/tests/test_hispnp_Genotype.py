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
            hps.Genotype(metadata=None,
                         probabilities=gen_modified.probabilities)
