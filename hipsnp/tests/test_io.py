import tempfile
import datalad.api as dl
import pytest
import shutil

import hipsnp as hps


def test_from_bgen_multiple_identical_files():
    """The same file is passed multiple times to read_bgen"""
    nFiles = 5
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'
    with tempfile.TemporaryDirectory() as tmpdir:
        print(tmpdir + '/')
        dataset = dl.install(source=source, path=tmpdir + '/')  # type: ignore
        dataset.get()
        bgenfile = [tmpdir + '/imputation/example_c1_v0.bgen']
        bgenfile *= nFiles

        with pytest.raises(ValueError, match='duplicated bgen files'):
            hps.read_bgen(files=bgenfile)


def test_from_bgen_files_duplicate_rsid():
    """Copy and rename a mock file to have variaous files with same content
    Duplicated RSIDs should be ignored"""
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = dl.install(source=source, path=tmpdir + '/')  # type: ignore
        dataset.get()
        bgenfile = tmpdir + '/imputation/example_c1_v0.bgen'
        bgenfile2 = tmpdir + '/imputation/example2_c1_v0.bgen'
        shutil.copy(bgenfile, bgenfile2)

        gen_ref = hps.read_bgen(files=bgenfile)

        with pytest.warns(RuntimeWarning, match='duplicated RSIDs'):
            gen_dup = hps.read_bgen(files=[bgenfile, bgenfile2])

        hps.utils.testing.assert_genotype_equal(gen_ref, gen_dup)


def test_from_bgen_non_existing_file():
    with pytest.raises(FileNotFoundError, match='file does not exist'):
        hps.read_bgen(files='/nonexisting/this.bgen')
