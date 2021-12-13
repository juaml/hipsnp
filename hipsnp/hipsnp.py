import os
import shutil
import requests
import pandas as pd
import numpy as np
from datalad import api as datalad
from bgen_reader import open_bgen
from pathlib import Path
import copy
from functools import reduce

from . utils import warn, raise_error, logger


def get_chromosome_data(c, datalad_source='ria+http://ukb.ds.inm7.de#~genetic',
                        imputation_dir='imputation', data_dir='/tmp/genetic'):
    """Get a particular chromosome's (imputed) data from a datalad dataset or
    a directory

    Parameters
    ----------
    c : str
        Chormosome number
    datalad_source : str or None, optional
        datalad data source. if None, the directory of chromosome data should
        be given with data_dir and imputation_dir arguments
        (i.e. <data_dir>/<imputationdir>'), by default
        'ria+http://ukb.ds.inm7.de#~genetic'
    imputationdir : str, optional
         directory in which the imputation files are stored,
         by default 'imputation'.
    data_dir : str, optional
        directory to use for the datalad dataset, or directory to actual
        data files, by default '/tmp/genetic'. Chromosome data will or should
        be placed on '<data_dir>/<imputationdir>'

    Returns
    -------
    files: list of str
        path to file names
    ds: datalad Dataset or None
        datalad dataset object with chromome data, None if data is in a
        directory
    getout : list of datalad get outputs or "datalad not used"

    """
    data_dir = Path(data_dir)

    if datalad_source:
        ds = datalad.install(source=datalad_source, path=data_dir)  # type: ignore
        files = list(data_dir.joinpath(imputation_dir).glob(f'*_c{c}_*'))
        getout = ds.get(files)
    else:
        ds = None
        files = list(data_dir.joinpath(imputation_dir).glob(f'*_c{c}_*'))
        getout = ['datalad not used'] * len(files)

    if len(files) == 0:
        raise_error(f'No files were found on disk for chromosome {c}')

    return files, ds, getout


def request_ensembl_rsid(rsid):
    """Make a REST call to ensembl.org and return a JSON object with
    the information of the variant of given a rsid

    Parameters
    ----------
    rsid : str
        rsid starting with 'rs'

    Returns
    -------
    JSON object

    Raises
    ------
    ValueError
        The rsid provided is not valid
    """    """"""

    """
    make a REST call to ensemble and return json info of a variant given a rsid
    rsid: string
    returns: json object
    """
    if not isinstance(rsid, str) or rsid[0:2] != 'rs':
        raise_error(f'rsid must be a string with "rs"')

    url = f'http://rest.ensembl.org/variation/human/{rsid}' \
        '?content-type=application/json'
    response = requests.get(url)
    return response.json()


def rsid_chromosome_DataFrame(rsids, chromosomes=None):
    """Build pandas DataFrame with rsids and chormosomes. If chormoseomes are
    not given they will be retrieved from ensembl.org for each rsids.

    Parameters
    ----------
    rsids : str or list of str
        rsids, list of rsids, or path to tab separated csv file with rsids or 
        PGS file
    chromosomes : None, str or list of str, optional
        list of chromosomes, by default None and retrieves the chromosome from
        ensemble.org

    Returns
    -------
    pandas DataFrame
        dataframe with columns 'rsids' and 'chromosomes'
    """
    if isinstance(rsids, str) and Path(rsids).is_file():
        rsids = pd.read_csv(rsids, header=None, sep='\t', comment='#')
        if rsids.shape[1] > 1:
            # this check provides support for PGS files
            if isinstance(rsids.iloc[0, 1], str):
                rsids.drop(index=0, inplace=True)
            chromosomes = list(rsids.iloc[:, 1])  # .astype('str') ?
            chromosomes = [str(c) for c in chromosomes]
        rsids = list(rsids.iloc[:, 0])
    elif isinstance(rsids, str):
        rsids = [rsids]

    if chromosomes is None:
        # get from ensembl
        chromosomes = [None] * len(rsids)
        for rs in range(len(rsids)):
            ens = request_ensembl_rsid(rsids[rs])
            ens = ens['mappings']
            for m in range(len(ens)):
                if ens[m]['ancestral_allele'] is not None:
                    chromosomes[rs] = ens[m]['seq_region_name']
    else:
        if len(chromosomes) != len(rsids):
            raise_error(f'Different amount of rsids {len(rsids)} \
                          and chromosomes {len(chromosomes)}')

        if isinstance(chromosomes, str) or isinstance(chromosomes, int):
            chromosomes = [chromosomes]
        chromosomes = [str(c) for c in chromosomes]

    df = pd.DataFrame()
    df['chromosomes'] = chromosomes
    df['rsids'] = rsids
    return df


def pruned_bgen_from_Datalad(
        rsids, outdir,
        datalad_source="ria+http://ukb.ds.inm7.de#~genetic",
        qctool=None, datalad_drop=True, datalad_drop_if_got=True,
        data_dir=None, recompute=False, chromosomes=None):
    """ Creates a .bgen format file with <rsids> and <chormosomes> 
    frome a datalad Dataset. .bgen file is created with qctool v2 (See notes)

    Parameters
    ----------
    rsids : str or list of str
        list of rsids 
    outdir : str
        path to save bgen files and csv file with used rsids
    datalad_source : str, optional
        datalad data source, by default "ria+http://ukb.ds.inm7.de#~genetic"
    qctool : str, optional
        path to qctool, by default None
    datalad_drop : bool, optional
         whether to drop the datalad dataset after getting the files, 
         by default True
    datalad_drop_if_got : bool, optional
        whether to drop files only if downloaded with get, by default True
    data_dir : str, optional
        directory to use for the (datalad) dataset, by default None
    recompute : bool, optional
        whether to recompute re-calculation (based on output file presence),
        by default False
    chromosomes : list of str, optional
        list of chromosomes to process, by default None which uses all
        chromosomes. There should be one chromosome of each rsid provided.

    Returns
    -------
    pandas DataFrame
        pandas dataframe with rsid-chromosome pairs

    Notes
    -----
    qctool must be installed (see https://www.well.ox.ac.uk/~gav/qctool/)

    """ 
    # check if qctool is available
    if qctool is None or Path(qctool).is_file() is False:
        qctool = shutil.which('qctool')
        if qctool is None:
            raise_error(f'qctool cannot be found')

    if not Path(outdir).exists():
        Path(outdir).mkdir()
    if recompute is True and list(Path(outdir).iterdir()):
        raise_error(f'the output directory must be empty')

    # get chromosome of each rsid
    if chromosomes is not None and len(chromosomes) != len(rsids):
        raise_error('Mismatch between the number of chrmosomes and rsids')

    ch_rs = rsid_chromosome_DataFrame(rsids, chromosomes=chromosomes)

    uchromosomes = ch_rs['chromosomes'].unique()
    files = None
    ds = None
    logger.info(f'Chromosomes needed: {uchromosomes}')
    for ch in enumerate(uchromosomes):

        file_out = Path(outdir, 'chromosome' + str(ch) + '.bgen')

        if recompute is False and file_out.is_file():
            warn(f'chromosome {ch} output file exists. It will not be \
                  recomputed. Skipping file: {file_out}')
            continue

        rs_ch = [rsids[i] for i, ch_x in enumerate(ch_rs['chromosomes'])
                 if ch_x == ch]
        if len(rs_ch) == 0:
            warn(f'Chromosome {ch} not matching list of chromosomes,\
                   skipping it')
            continue

        if len(rs_ch) < 11:
            logger.info(f'rsids: {rs_ch}\n')

        # get the data
        files, ds, getout = \
            get_chromosome_data(ch,
                                datalad_source=datalad_source,
                                data_dir=data_dir)
        for f_ix, getout_val in enumerate(getout):
            status = getout_val['status']
            if status != 'ok' and status != 'notneeded':
                ds.remove(dataset=data_dir)
                raise_error(f'datalad: error getting file {f_ix}: \
                              {getout_val["path"]} \n') 
            else:
                logger.info(f'datalad: status {status} file {files[f_ix]}')

        # find the bgen and sample files
        file_bgen = []
        file_sample = []
        for fl in files:
            name, ext = os.path.splitext(fl)
            if ext == '.bgen':
                file_bgen.append(fl)
            elif ext == '.sample':
                file_sample.append(fl)
        if len(file_bgen) != 1 or len(file_sample) != 1:
            raise_error(f'Wrong bgen and/or sample files for chromosome {ch}')
    
        file_rsids = Path(outdir, 'rsids_chromosome' + ch + '.txt')
        df = pd.DataFrame(rs_ch)
        df.to_csv(file_rsids, index=False, header=False)

        cmd = (qctool + ' -g ' + str(file_bgen[0]) + ' -s ' + str(file_sample[0])
               + ' -incl-rsids ' + str(file_rsids)  + ' -og ' + str(file_out)
               + ' -ofiletype bgen_v1.2 -bgen-bits 8')
        # from a .bgen and a .samples file, make a .bgen file with rsids
        # in -incl-rsids in directory file_out

        logger.info('running qctool: {cmd}\n')
        os.system(cmd)

        if datalad_drop:
            common_prefix = os.path.commonprefix([files[0], ds.path])
            files_rel = [os.path.relpath(path, common_prefix)
                         for path in files]
            if datalad_drop_if_got:
                for fi in range(len(getout)):
                    if (getout[fi]['status'] == 'ok' and
                            getout[fi]['type'] == 'file'):
                        logger.info(f'datalad: dropping file {files_rel[fi]}')
                        ds.drop(files_rel[fi])
            else:
                logger.info(f'datalad: dropping all files\n')
                ds.drop(files_rel)
    return ch_rs, files, ds


def read_weights(weights):
    """read weights from a .csv or .pgs file. Minimaly, csv headers 
        should contain "effect_allele" or "ea", "effect_weight" or ""weight", 
        "snpid" or "rsid", and "chr_name" or "chr"

    Parameters
    ----------
    weiths : str
        Path to the csv or pgs file with the weigths. 

    Returns
    -------
    DataFrame:
        weigths by RSID

    Rasises
    -------
    ValueErthe CSV does not contain required fields or infromation is worng
    """
    try:
        weights = pd.read_csv(weights, sep='\t', comment='#',
                              converters={'effect_allele': np.char.upper})
        weights.columns = [x.lower() for x in weights.columns]
        weights.rename(columns={'snpid': 'rsid', 'chr_name': 'chr',
                                'effect_allele': 'ea',
                                'effect_weight': 'weight'},
                       inplace=True)

        weights.set_index('rsid', inplace=True)

    except ValueError as e:
        raise_error(f'Fails reading weights', klass=e)

    if 'ea' not in weights.columns or 'weight' not in weights.columns:
        raise_error(f'Weights contains wrong column names')

    if np.sum(weights.index.duplicated()) != 0:
        weights = weights.loc[~weights.index.duplicated()]
        warn(f'Weights" has duplicated RSIDs, only the first, '
              'appearane will be kept')

    if not np.isin(weights['ea'], ['A', 'C', 'T', 'G']).all():
        raise_error(f'effect allelel in weights is not "A", "C", "T", or "G"')

    return weights


class Genotype():
    """Genotype class. Models a genotype including the list of chromosomes,
    positions, and the probabilities of each allele in each sample.

    Attributes
    ----------
    _metadata: pd.DataFrame
        Metadata of the genotype. The dataframe should contain the
        columns RSIDS (as index), CHROM, POS, ID and FORMAT.
        Each RSID should be unique and map one-to-one to a chromosome.
    _probabilities: dict(str : tuple(np.array of size (n_samples),
                                     np.array of size (n_samples, 3))))
        Probabilities of each allele in each sample, for each chromosome.
        Each value is a tuple with the samples and a 2D numpy array with
        the probabilites of each allele in each sample
        (REF-REF, ALT-REF, and ALT-ALT).
    _consolidated: bool
        If consolidated, the object contains data in which all RSIDs have the
        same samples and in the same order.

    """
    def __init__(self, metadata, probabilities):
        """
        Genotype constructor

        Parameters
        ----------
        metadata: pd.DataFrame
            Metadata of the genotype. The dataframe should contain the
            columns RSIDS (as index), CHROM, POS, ID and FORMAT.
            Each RSID should be unique and map one-to-one to a chromosome.
        probabilities: dict(str : tuple(str, np.array of size (n_samples, 3))))
            Probabilities of each allele in each sample, for each chromosome.
            Each value is a tuple with the samples and a 2D numpy array with
            the probabilites of each allele in each sample
            (REF-REF, ALT-REF, and ALT-ALT).
        """
        self._validate_arguments(metadata, probabilities)
        self._metadata = metadata
        self._probabilities = probabilities
        self._consolidated = False

    def _clone(self):
        """Clone the object"""
        out = Genotype(
            self._metadata.copy(), copy.deepcopy(self._probabilities))
        out._consolidated = self._consolidated
        return out

    @property
    def rsids(self):
        """ RSIDs present in the genotype.

        Returns
        -------
        list(str)
            The rsids present in the genotype.
        """
        return list(self._metadata.index)

    @property
    def is_consolidated(self):
        """ If consolidated, the object contains data in which all RSIDs have
        the same samples and in the same order.

        Returns
        -------
        bool
            True if the genotype is consolidated, False if otherwise.
        """
        return self._consolidated

    @property
    def metadata(self):
        """ Metadata of the genotype, including the columns RSIDS (as index),
        CHROM, POS, ID and FORMAT.

        Returns
        -------
        pd.DataFrame
            the Metadata of the genotype.
        """
        return self._metadata

    @property
    def probabilities(self):
        """ Probability of each allele combination foir each sample of an rsids

        Returns
        -------
        dict(str : tuple(str, np.array of size (n_samples, 3))))
            The three probalities associated to the allele combinations
            REF-REF, ALT-REF, and ALT-ALT for each sample of each rsids.
        """
        return self._probabilities

    def filter(self, rsids=None, samples=None, weights=None, inplace=True):
        """Filter the genotype by rsids and samples. Alternatively, a weights
        definition (csv or DataFrame) can be provided. The rsids will then be
        extracted from this file.

        Parameters
        ----------
        rsids : str or list of str | None
            RSIDs to keep. If None, does not filter (default).
        samples : str or list of str | None
            Samples to keep. If None, does not filter (default).
        weights : str, Path or DataFrame | None
            Path to a CSV or PSG file with the weights,
            or pandas DataFrame with weights as provided by `read_weights`
        inplace: bool
            If true, modifes the object in place (default). If False, returns
            a new object without modifying the original one.

        Returns
        -------
        Genotype:
            The filtered genotype object.

        Rasises
        -------
        ValueError
            If the filtered data is empty.
        """
        if rsids is None and samples is None and weights is None:
            warn(f'Nothing to filter')
            if inplace:
                return self
            else:
                return self._clone()

        if isinstance(rsids, str):
            rsids = [rsids]

        # Check if we need to handle weights
        if isinstance(weights, str) or isinstance(weights, Path):
            weights = read_weights(weights)

        if weights is not None:
            rsids_weights = weights.index.to_list()  # type: ignore
            if rsids is None:
                rsids = rsids_weights
            else:
                rsids = list(np.intersect1d(rsids, rsids_weights))

        out = self._filter_by_rsids(
            rsids=rsids, inplace=inplace)._filter_by_samples(
                samples=samples, inplace=inplace)

        return out

    def _filter_by_samples(self, samples=None, inplace=True):
        """Filter object by samples

        Parameters
        ----------
        samples : str or list of str | None
            Samples to keep. If None, does not filter (default).
        inplace: bool
            If true, modifes the object in place (default). If false, returns
            a new object without modifying the original one.

        Returns
        -------
        Genotype:
            The filtered genotype object.

        Raises
        -------
        ValueError
            If the filtered data is empty.
        """
        if samples is None:
            if inplace is True:
                return self
            else:
                return self._clone()

        if not isinstance(samples, list):
            samples = [samples]

        probs_filtered = dict()
        # Iterate over all probabilities and keep only the selected samples
        for rsid, s_prob in self.probabilities.items():
            mask_samples = np.isin(s_prob[0], samples)

            # check that there is at least one sample for that rsid
            if mask_samples.any():
                probs_filtered[rsid] = (
                    s_prob[0][mask_samples],  # samples
                    s_prob[1][mask_samples, :])  # probabilities

        reamining_rsids = list(probs_filtered.keys())
        if len(reamining_rsids) == 0:
            raise_error(f'No samples matching filter specifications')

        # Filter metadata to keep only rsids with samples
        meta_filtered = self.metadata.filter(items=reamining_rsids, axis=0)

        if inplace:
            self._probabilities = probs_filtered
            self._metadata = meta_filtered
            return self
        else:
            out = Genotype(metadata=meta_filtered,
                           probabilities=probs_filtered)
            return out

    def _filter_by_rsids(self, rsids=None, inplace=True):
        """Filter Genotype data object by RSID

        Parameters
        ----------
        samples : str or list of str | None
            RSIDs to keep. If None, does not filter (default).
        inplace: bool
            If true, modifes the object in place (default). If false, returns
            a new object without modifying the original one.

        Returns
        -------
        Genotype:
            The filtered genotype object.

        Raises
        -------
        ValueError
            If the filtered data is empty.
        """
        if rsids is None:
            if inplace is True:
                return self
            else:
                return self._clone()

        if not isinstance(rsids, list):
            rsids = [rsids]

        meta_filtered = self.metadata.filter(items=rsids, axis=0)
        if meta_filtered.empty:
            raise_error(f'No RSIDs matching filter specifications')

        probs_filtered = {k_rsid: self.probabilities[k_rsid]
                          for k_rsid in rsids if k_rsid in self.rsids}

        if inplace:
            self._probabilities = probs_filtered
            self._metadata = meta_filtered
            return self
        else:
            out = Genotype(metadata=meta_filtered,
                           probabilities=probs_filtered)
            return out

    def consolidate(self, inplace=True):
        """Align samples consistently across all RSIDs. If a sample is not
        found in all RSID, the sample is discarded.

        Parameters
        ---------_
        inplace: bool
            If true, modifes the object in place (default). If false, returns
            a new object without modifying the original one.

        Returns
        -------
        Genotype:
            The consolidated genotype object.

        Raises
        -------
        ValueError
            If there are no common samples across RSIDs
        """
        # find common samples across all RSIDs
        common_samples = reduce(
            np.intersect1d, (sp[0] for sp in self.probabilities.values()))
        if len(common_samples) == 0:
            raise_error('There are no common samples across all RSIDs')

        consol_prob_dict = {}
        for rsid, sample_prob in self.probabilities.items():
            # Get index of common_samples appearing on other RSIDs
            _, _, consol_idx = np.intersect1d(
                common_samples, sample_prob[0], assume_unique=True,
                return_indices=True)
            consol_prob_dict[rsid] = (
                common_samples, sample_prob[1][consol_idx, :])

        if inplace:
            self._probabilities = consol_prob_dict
            self._consolidated = True
            return self
        else:
            out = Genotype(metadata=self.metadata,
                           probabilities=consol_prob_dict)
            out._consolidated = True
            return out

    def _consolidated_samples(self):
        """List of samples present in a consolidated genotype.

        Returns
        -------
        list of strings
            samples in the consolidated genotype.

        Raises
        -------
        ValueError
            If the object is not consolidated
        """
        if not self.is_consolidated:
            raise_error(
                'Samples are not consolidated across RSIDs. '
                'Execute `consolidate` first.')

        uniq_samples = self.probabilities[self.rsids[0]][0]
        return uniq_samples


    def _consolidated_probabilites(self):
        """ Return the probabilities of the three allele
        combinations for all RSIDs and samples as a 3D numpy array.
        This is possible only in consolidated objects.

        Retruns
        -------
        np.array (n_samples, n_rsids, 3)
            The probabilities of the three allele combinations for all rsids
            and samples.
        """
        if not self.is_consolidated:
            raise_error('Samples are not consolidated across RSIDs. '
                        'Execute `consolidate` first.')

        n_rsids = len(self.probabilities)
        n_samples = len(self._consolidated_samples())
        consol_prob_matrix = np.zeros((n_rsids, n_samples,  3))

        for i, sample_proba in enumerate(self.probabilities.values()):
            consol_prob_matrix[i, :, :] = sample_proba[1]
        return consol_prob_matrix

    def alleles(self, rsids=None, samples=None):
        """Get the alleles for this genotype object.

        Parameters
        ----------
        rsids : list of str, optional
            rsids to be used, by default None
        samples : list of str, optional
            Samples to be used, by default None

        Returns
        -------
        alleles: pandas DataFrame
            Most probable alleles of each rsid and sample.
        alleles_idx : pandas DataFrame
            Indexes of the most probable allele 0=REFREF, 1=ALTREF, 2=ALTALT
        """
        gen_filt = self.filter(samples=samples, rsids=rsids, inplace=False)

        if not gen_filt.is_consolidated:
            logger.info(
                'Samples are not consolidated across RSIDs. Consolidating...')
            gen_filt.consolidate(inplace=True)

        probs = gen_filt._consolidated_probabilites()

        n_rsids = len(gen_filt.rsids)
        n_samples = len(gen_filt._consolidated_samples())

        logger.info(f'Calculating genotypes for {n_rsids} SNPs and \
                    {n_samples} samples ... ')

        genotype_allele = np.empty((n_rsids, n_samples), dtype=object)
        genotype_012 = np.zeros((n_rsids, n_samples), dtype=int)

        # reshape to allow for straight indexing
        ref = np.tile(gen_filt.metadata['REF'].to_numpy(), (n_samples, 1)).T
        alt = np.tile(gen_filt.metadata['ALT'].to_numpy(), (n_samples, 1)).T

        i_max_p = np.argmax(probs, axis=2)

        # Sort needs a single array, but to add characters it needs two arrays
        tmp = np.split(
            np.sort(
                np.vstack(
                    (ref[i_max_p == 1], alt[i_max_p == 1])).astype(str),
                axis=1),
            2, axis=0)
        g_allele = np.squeeze(np.char.add(tmp[0], tmp[1]))

        genotype_allele[i_max_p == 0] = ref[i_max_p == 0] + ref[i_max_p == 0]
        genotype_allele[i_max_p == 1] = g_allele
        genotype_allele[i_max_p == 2] = alt[i_max_p == 2] + alt[i_max_p == 2]

        genotype_012 = i_max_p

        genotype_allele = pd.DataFrame(
            data=genotype_allele, index=gen_filt.rsids,
            columns=gen_filt._consolidated_samples())
        genotype_012 = pd.DataFrame(
            data=genotype_012, index=gen_filt.rsids,
            columns=gen_filt._consolidated_samples())

        return genotype_allele, genotype_012

    def riskscores(self, weights, rsids=None, samples=None):
        """ Compute the risk score and dosage for this genotype object.

        Parameters
        ----------
        weights : str or pd.DataFrame,
            Path to CSV or PGS file with weights.
        rsids : list of str | None
            RSIDs to be used. If None (default), all RSIDs are used.
        samples : list of str | None
            Samples to be used. If None (default), all samples are used.

        Returns
        -------
        dosage : pd.DataFrame
            Dataframe with the dosage by rsid and samples
        riskscores : pd.DataFrame
            DataFrame with riskscores by samples
        """

        weights = read_weights(weights)

        gen_filt = self.filter(
            samples=samples, rsids=rsids, weights=weights, inplace=False)

        if not gen_filt.is_consolidated:
            gen_filt.consolidate(inplace=True)

        # sort all DataFrames by the RSIDS in gen_filt.probabilities
        rsids_as_in_prob = list(gen_filt.probabilities.keys())

        # TODO: This should not be needed here!
        gen_filt._metadata = gen_filt._metadata.reindex(rsids_as_in_prob)
        weights = weights.reindex(rsids_as_in_prob)  # type: ignore

        n_rsid = len(gen_filt.rsids)
        n_sample = len(gen_filt._consolidated_samples())

        logger.info(f'Calculating riskscores for {n_rsid} SNPs and \
                    {n_sample} samples ... ')

        ref = np.tile(gen_filt.metadata['REF'].to_numpy(), (n_sample, 1)).T
        alt = np.tile(gen_filt.metadata['ALT'].to_numpy(), (n_sample, 1)).T
        probs = gen_filt._consolidated_probabilites()

        ea = weights['ea'].to_numpy()
        ea = np.tile(ea, (n_sample, 1)).T

        # compute individual dosage
        mask_ea_eq_ref = ea == ref
        mask_ea_eq_alt = ea == alt

        dosage = np.zeros((n_rsid, n_sample))
        dosage[mask_ea_eq_ref] = (probs[mask_ea_eq_ref, 1]
                                  + 2 * probs[mask_ea_eq_ref, 0])
        dosage[mask_ea_eq_alt] = (probs[mask_ea_eq_alt, 1]
                                  + 2 * probs[mask_ea_eq_alt, 2])

        wSNP = weights['weight'].to_numpy().astype(float).reshape(n_rsid, 1)
        riskscores = np.sum(dosage * wSNP, axis=0)

        dosage = pd.DataFrame(
            data=dosage, columns=gen_filt._consolidated_samples(),
            index=gen_filt.rsids)
        riskscores = pd.DataFrame(
            data=riskscores, index=gen_filt._consolidated_samples())
        return dosage, riskscores

    @staticmethod
    def _from_bgen(files, verify_integrity=False):
        """Read bgen file. Return Genotype object with metadata and probabilites

        Parameters
        ----------
        files : str or list(str)
            Files to be read
        verify_integrity : bool
            If True, verify that there RSIDs are not repeated.
            Defaults to False due to performance reasons.

        Returns
        -------
        genotype : Genotype
            The genotype object as read from the files.
        """
        if isinstance(files, str):
            files = [files]

        if len(files) != len(set(files)):
            raise_error("There are duplicated bgen files")
        # make sure that files exist
        if not all([Path(f).is_file() for f in files]):
            raise_error('bgen file does not exist', FileNotFoundError)

        # read all the files
        logger.info(f'Reading {len(files)} bgen files...')

        metadata = pd.DataFrame()
        probabilites = dict()
        for f in files:
            logger.info(f'Reading {f}')
            with open_bgen(f, verbose=False) as bgen:
                # we can only deal with biallelic variants
                if np.any(bgen.nalleles != 2):
                    raise_error('Only biallelic variants are allowed')

                # find duplicate RSIDs within a file
                _, iX_unique_in_file = np.unique(bgen.rsids, return_index=True)
                if (iX_unique_in_file.shape[0] !=
                        bgen.rsids.shape[0]):  # type: ignore
                    warn(f'Duplicated RSIDs in file {f}')

                # find duplicates with previous files
                if (not metadata.empty and
                        np.sum(metadata.index == bgen.rsids) != 0):
                    warn(f'Files have duplicated RSIDs')
                    # indexes with rsids not previously taken
                    # to keep unique RSIDS
                    mask_unique_btwb_files = np.isin(
                        bgen.rsids, metadata.index, invert=True)
                    mask_to_keep = np.zeros(len(bgen.rsids), dtype=np.bool_)
                    mask_to_keep[iX_unique_in_file
                                 [mask_unique_btwb_files]] = True
                else:
                    mask_to_keep = np.ones(len(bgen.rsids), dtype=np.bool_)

                if any(mask_to_keep):

                    alleles = np.array(
                        [np.char.upper(val) for val in
                         np.char.split(
                             bgen.allele_ids[mask_to_keep], sep=',')])
                    if not np.isin(alleles, ['A', 'C', 'T', 'G']).all():
                        raise_error(
                            f'alleles not "A", "C", "T", or "G" in file {f}')

                    # dataframe with metadata of unique RSIDS.
                    tmp = pd.DataFrame(index=bgen.rsids[mask_to_keep])
                    tmp = tmp.assign(
                        REF=alleles[:, 0], ALT=alleles[:, 1],
                        CHROM=bgen.chromosomes[mask_to_keep],
                        POS=bgen.positions[mask_to_keep],
                        ID=bgen.ids[mask_to_keep], FORMAT='GP')

                    if f == files[0]:
                        myjoin = 'outer'
                    else:
                        myjoin = 'inner'
                    # concatenate metadata of files
                    # TODO: Concatenate only once at the end
                    metadata = pd.concat(
                        [metadata, tmp], join=myjoin, axis=0,
                        verify_integrity=verify_integrity)

                    # crear probabilites data dictionary
                    probs = bgen.read()
                    tmp_probabilites = {
                        k_rsid: (np.array(bgen.samples),
                                 np.squeeze(probs[:, i, :]))  # type: ignore
                        for i, k_rsid in enumerate(tmp.index)}
                    probabilites.update(tmp_probabilites)

        return Genotype(metadata, probabilites)

    @staticmethod
    def _validate_arguments(meta, prob):
        """Basic check of Genotype arguments
            * metadata has columns 'REF', 'ALT', 'CHROM', 'POS', 'ID', 'FORMAT'
            * same order of RSID in metadata as probabilities
            * probabilities has same dimensions
        Parameters
        ----------
        meta : pandas DataFrame
            Genotype.metadata atribute
        prob : dict of tuples with list of str and numpy array
            Genotype.probabilites atribute
        """
        if sum((col in ['REF', 'ALT', 'CHROM', 'POS', 'ID', 'FORMAT']
                for col in meta.columns)) < 6:
            raise_error("Missign columns in metadata")
        if sorted(meta.index) != sorted(prob.keys()):
            raise_error("Mismatch of RSIDs between metadata and probabilities")
        if any([len(prob[k_key][0]) != prob[k_key][1].shape[0] or
                prob[k_key][1].shape[1] != 3 for k_key in prob.keys()]):
            raise_error("Dimension mismatch between samples and probabilities")


def read_bgen(files, verify_integrity=False):
    """Read bgen files into a single Genotype object

        Parameters
        ----------
        files : str or list(str)
            Files to be read
        verify_integrity : bool
            If True, verify that there RSIDs are not repeated.
            Defaults to False due to performance reasons.

        Returns
        -------
        genotype : Genotype
            The genotype object as read from the files.
        """
    return Genotype._from_bgen(files, verify_integrity)