import os
import stat
import tempfile
from pathlib import Path
import datalad.api as dl
import pytest
import shutil
import pandas as pd
from pandas._testing import assert_frame_equal

import hipsnp as hps


def test_from_bgen_multiple_identical_files():
    """The same file is passed multiple times to read_bgen"""
    nFiles = 5
    source = 'https://gin.g-node.org/juaml/datalad-example-bgen'
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
    source = 'https://gin.g-node.org/juaml/datalad-example-bgen'
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


def test_read_weights():
    data_path = hps.utils.testing.get_testing_data_dir()
    wfile = data_path / 'weights_5_duplicatedRSID.csv'
    with pytest.warns(RuntimeWarning, match='duplicated RSIDs'):
        w = hps.read_weights(wfile)
    assert w.shape[0] == 4 and w.shape[1] == 3
    assert sorted(list(w.index)) == sorted(
        ['RSID_2', 'RSID_5', 'RSID_6', 'RSID_7'])

    wfile =  data_path / 'weights_5_other_headers.csv'
    with pytest.raises(ValueError, match='wrong column names'):
        hps.read_weights(wfile)


def test_get_chromosomes():
    """Test getting chromosomes from RSIDs"""
    rsidsFail = ['RS699', 'ID699', '699']
    with pytest.raises(ValueError, match='must start with "rs"'):
        hps.get_chromosomes_from_ensembl(rsidsFail)

    rsidsFail = [100]
    with pytest.raises(ValueError, match='must be a list of strings'):
        hps.get_chromosomes_from_ensembl(rsidsFail)

    rsidsFail = ['rs699132213']
    with pytest.raises(ValueError, match='rs699132213 not found for human'):
        hps.get_chromosomes_from_ensembl(rsidsFail)

    data_path = hps.utils.testing.get_testing_data_dir()
    rsidFile =  data_path / 'rsid_699_102.csv'
    rsid = ['rs699', 'rs102']
    out_str = hps.get_chromosomes(rsid)
    out_f = hps.get_chromosomes(rsidFile)
    assert_frame_equal(out_str, out_f)

    pgsfile = data_path / 'weights_PGS000001.txt'
    out_pgs = hps.get_chromosomes(pgsfile)

    assert isinstance(out_pgs, pd.DataFrame)  # type: ignore
    assert out_pgs.shape[0] == 77


def test_get_qctool():
    """Test for QCTOOL executable"""
    old_path = os.environ['PATH']
    os.environ['PATH'] = ''
    with pytest.raises(ValueError, match='qctool not found'):
        hps.utils.get_qctool()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        fname = tmpdir / 'qctool'
        os.environ['PATH'] = tmpdir.as_posix()
        with open(fname, 'w') as f:
            f.write('#!/bin/bash\n')
        mode = os.stat(fname).st_mode
        mode |= stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        os.chmod(fname, mode)
        qctool = hps.utils.get_qctool()
        assert qctool == 'qctool'

        os.environ['PATH'] = ''
        os.environ['QCTOOL_PATH'] = fname.as_posix()
        qctool = hps.utils.get_qctool()
        assert qctool == fname.as_posix()

        os.environ['QCTOOL_PATH'] = ''
        with pytest.raises(ValueError, match='QCTOOL_PATH has a wrong '):
            hps.utils.get_qctool()

        wrong_fname = tmpdir / 'qctool2'
        with open(wrong_fname, 'w') as f:
            f.write('#!/bin/bash\n')
        os.environ['QCTOOL_PATH'] = wrong_fname.as_posix()
        with pytest.raises(ValueError, match='The executable in '):
            hps.utils.get_qctool()

    os.environ['PATH'] = old_path


def test_genotype_from_datalad():

    source = 'https://gin.g-node.org/juaml/datalad-example-bgen'
    rsids = ['RSID_2', 'RSID_3', 'RSID_5', 'RSID_6', 'RSID_7']
    chromosomes = ['1'] * len(rsids)
    with tempfile.TemporaryDirectory() as datadir:
        with tempfile.TemporaryDirectory() as workdir:
            # Get the dataset and cleanup
            genotype = hps.genotype_from_datalad(
                rsids, workdir=workdir, datadir=datadir,
                chromosomes=chromosomes, datalad_source=source,
                recompute=False)
            assert (Path(workdir) / 'chromosome1.bgen').exists()
        assert genotype.rsids == rsids
        assert not (
            Path(datadir) / 'imputation' / 'example_c1_v0.sample').exists()
        assert not (
            Path(datadir) / 'imputation' / 'example_c1_v0.bgen').exists()

        with tempfile.TemporaryDirectory() as workdir:
            # Get the dataset and do not cleanup
            genotype = hps.genotype_from_datalad(
                rsids, workdir=workdir, datadir=datadir,
                chromosomes=chromosomes, datalad_source=source,
                recompute=True, datalad_drop=False)
        assert genotype.rsids == rsids
        assert (Path(datadir) / 'imputation' / 'example_c1_v0.sample').exists()
        assert (Path(datadir) / 'imputation' / 'example_c1_v0.bgen').exists()

        # Get the dataset and cleanup only got files
        with tempfile.TemporaryDirectory() as workdir:
            genotype = hps.genotype_from_datalad(
                rsids, workdir=workdir, datadir=datadir,
                chromosomes=chromosomes, datalad_source=source,
                recompute=True)
            assert genotype.rsids == rsids
        assert (Path(datadir) / 'imputation' / 'example_c1_v0.sample').exists()
        assert (Path(datadir) / 'imputation' / 'example_c1_v0.bgen').exists()

        # Get the dataset and cleanup everything
        with tempfile.TemporaryDirectory() as workdir:
            genotype = hps.genotype_from_datalad(
                rsids, workdir=workdir, datadir=datadir,
                chromosomes=chromosomes, datalad_source=source,
                recompute=True, datalad_drop=True)
        assert genotype.rsids == rsids
        assert not (
            Path(datadir) / 'imputation' / 'example_c1_v0.sample').exists()
        assert not (
            Path(datadir) / 'imputation' / 'example_c1_v0.bgen').exists()

        # Force recompute without empty workdir
        with tempfile.TemporaryDirectory() as workdir:
            genotype = hps.genotype_from_datalad(
                rsids, workdir=workdir, datadir=datadir,
                chromosomes=chromosomes, datalad_source=source,
                recompute=False)

            with pytest.raises(ValueError, match='must be empty'):
                genotype = hps.genotype_from_datalad(
                    rsids, workdir=workdir, datadir=datadir,
                    chromosomes=chromosomes, datalad_source=source,
                    recompute=True)
