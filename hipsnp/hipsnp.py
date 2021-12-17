import subprocess
import requests
import pandas as pd
import numpy as np
import time
from datalad import api as datalad
from bgen_reader import open_bgen
from pathlib import Path
import copy
from functools import reduce

from . utils import warn, raise_error, logger, get_qctool


class Genotype():
    """Genotype class. Models a genotype including the list of chromosomes,
    positions, and the probabilities of each allele in each sample.

    Attributes
    ----------
    _metadata: pd.DataFrame
        Metadata of the genotype. The dataframe should contain the
        columns RSIDS (as index), CHROM, POS, ID and FORMAT.
        Each RSID should be unique and map one-to-one to a chromosome.
    _probabilities: dict(str : tuple(np.array of size (n_samples), \
np.array of size (n_samples, 3))))
        Probabilities of each allele in each sample, for each chromosome.
        Each value is a tuple with the samples and a 2D numpy array with
        the probabilities of each allele in each sample
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
            the probabilities of each allele in each sample
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
            If true, modifies the object in place (default). If False, returns
            a new object without modifying the original one.

        Returns
        -------
        Genotype:
            The filtered genotype object.

        Raises
        ------
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
            If true, modifies the object in place (default). If false, returns
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
            If true, modifies the object in place (default). If false, returns
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
        ----------
        inplace: bool
            If true, modifies the object in place (default). If false, returns
            a new object without modifying the original one.

        Returns
        -------
        Genotype
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

    def _consolidated_probabilities(self):
        """ Return the probabilities of the three allele
        combinations for all RSIDs and samples as a 3D numpy array.
        This is possible only in consolidated objects.

        Returns
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
        if rsids is not None:
            gen_filt = self.filter(samples=samples, rsids=rsids, inplace=False)
        else:
            gen_filt = self._clone()

        if not gen_filt.is_consolidated:
            logger.info(
                'Samples are not consolidated across RSIDs. Consolidating...')
            gen_filt.consolidate(inplace=True)

        probs = gen_filt._consolidated_probabilities()

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
                axis=0),
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

        if not isinstance(weights, pd.DataFrame):
            weights = read_weights(weights)

        if rsids is not None or samples is not None or weights is not None:
            gen_filt = self.filter(
                samples=samples, rsids=rsids, weights=weights, inplace=False)
        else:
            gen_filt = self._clone()

        if not gen_filt.is_consolidated:
            logger.info(
                'Samples are not consolidated across RSIDs. Consolidating...')
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
        probs = gen_filt._consolidated_probabilities()

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
        """Read bgen file. Return Genotype object with metadata and
        probabilities

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

        metadata = None
        probabilities = dict()
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
                if (metadata is not None and
                        any(x in metadata.index for x in bgen.rsids)):
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

                    # concatenate metadata of files
                    # TODO: Concatenate only once at the end
                    if metadata is None:
                        metadata = tmp
                    else:
                        metadata = pd.concat(
                            [metadata, tmp], axis=0,
                            verify_integrity=verify_integrity)

                    # crear probabilities data dictionary
                    probs = bgen.read()
                    tmp_probabilities = {
                        k_rsid: (np.array(bgen.samples),
                                 np.squeeze(probs[:, i, :]))  # type: ignore
                        for i, k_rsid in enumerate(tmp.index)}
                    probabilities.update(tmp_probabilities)
        return Genotype(metadata, probabilities)

    @staticmethod
    def _validate_arguments(meta, prob):
        """Basic check of Genotype arguments
            * metadata has columns 'REF', 'ALT', 'CHROM', 'POS', 'ID', 'FORMAT'
            * same order of RSID in metadata as probabilities
            * probabilities has same dimensions

        Parameters
        ----------
        meta : pandas DataFrame
            Genotype.metadata attribute
        prob : dict of tuples with list of str and numpy array
            Genotype.probabilities attribute
        """
        if sum((col in ['REF', 'ALT', 'CHROM', 'POS', 'ID', 'FORMAT']
                for col in meta.columns)) < 6:
            raise_error("Missing columns in metadata")
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


_valid_chromosomes = [f'{x}' for x in range(1, 23)] + ['X', 'Y']


def get_chromosomes_from_ensembl(rsids, max_retries=5):
    """Make a REST call to ensembl.org and return a JSON object with
    the information of the variant of given a rsid

    Parameters
    ----------
    rsid : str or list of str
        RSID(s) to be queried. Must start with "rs"

    Returns
    -------
    list of str
        The chromosome(s) of the given rsid(s)

    Raises
    ------
    ValueError
        If the RSIDs provided are not valid
    """
    if not isinstance(rsids, list):
        rsids = [rsids]

    if not all([isinstance(rsid, str) for rsid in rsids]):
        raise_error('rsids must be a list of strings')

    if not all([rsid.startswith('rs') for rsid in rsids]):
        raise_error('rsids must start with "rs"')

    logger.info('Getting list from chromosomes from ensembl.org')
    chromosomes = []
    for rsid in rsids:
        logger.info(f'Getting chromosome for {rsid} from ensembl.org')
        t_chromosome = None
        url = (f'http://rest.ensembl.org/variation/human/{rsid}'
               '?content-type=application/json')
        ok = False
        response = None
        while ok is False and max_retries > 0:
            response = requests.get(url)
            if response.ok is True:
                ok = True
            else:
                if response.status_code == 304:
                    warn(f'ensembl.org replied with {response.status_code}. '
                         'Waiting 1 second and retrying.')
                    max_retries -= 1
                    time.sleep(1)
                else:
                    json_dict = response.json()
                    error = json_dict.get('error', 'Uknownw')
                    raise_error(
                        f'ensembl.org replied with {response.status_code}. '
                        f'Error: {error}')
        if max_retries == 0 and ok is False:
            raise_error('Could not get the rsid information from ensembl.org')
        json_dict = response.json()  # type: ignore
        if 'error' in json_dict:
            raise_error(
                'Error getting the chromosomes from ensembl.org: '
                f'{json_dict["error"]}')
        mappings = json_dict['mappings']
        for mapping in mappings:
            t_chromosome = mapping['seq_region_name']
            break

        if t_chromosome is None:
            raise_error(f'Chromosome for {rsid} not found')
        if t_chromosome not in _valid_chromosomes:
            raise_error(f'Chromosome {t_chromosome} not valid')
        chromosomes.append(t_chromosome)

    return chromosomes


def get_chromosomes(rsids, chromosomes=None):
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
    pd.DataFrame
        dataframe with columns 'rsids' and 'chromosomes'
    """
    if ((isinstance(rsids, str) and Path(rsids).is_file()) or
            isinstance(rsids, Path)):
        rsids = pd.read_csv(
            rsids, header=None, sep='\t', comment='#')  # type: ignore
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
        chromosomes = get_chromosomes_from_ensembl(rsids)
    else:
        if len(chromosomes) != len(rsids):
            raise_error(
                f'The number of RSIDs {len(rsids)} does not match the number '
                f'of chromosomes {len(chromosomes)}.')

        if isinstance(chromosomes, str) or isinstance(chromosomes, int):
            chromosomes = [chromosomes]
        chromosomes = [str(c) for c in chromosomes]

    df = pd.DataFrame({'rsids': rsids, 'chromosomes': chromosomes})
    return df.drop_duplicates()


def read_weights(fname, sep='\t'):
    """Read weights from a CSV/TSV or PGS file (see
    https://www.pgscatalog.org/downloads/#dl_ftp_scoring).

    Table headers must contain:
        * `effect_allele` or `ea`
        * `effect_weight` or `weight`
        * `snpid` or `rsid`
        * `chr_name` or `chr`

    Parameters
    ----------
    fname : str
        Path to the CSV/TSV or PGS file with the weights.

    Returns
    -------
    pd.DataFrame
        weights by RSID

    Raises
    ------
    ValueError
        If the file does not exist or the CSV does not contain required
        fields.
    """

    if isinstance(fname, str):
        fname = Path(fname)

    if not fname.exists():
        raise_error(f'File {fname.as_posix()} does not exist')

    weights = pd.read_csv(
        fname, sep=sep, comment='#',
        converters={'effect_allele': np.char.upper})

    weights.columns = [x.lower() for x in weights.columns]
    weights.rename(
        columns={'snpid': 'rsid', 'chr_name': 'chr', 'effect_allele': 'ea',
                 'effect_weight': 'weight'}, inplace=True)

    weights.set_index('rsid', inplace=True)

    if 'ea' not in weights.columns or 'weight' not in weights.columns:
        raise_error(f'File {fname.as_posix()} contains wrong column names')

    duplicated = weights.index.duplicated(keep='first')
    if np.any(duplicated):
        weights = weights.loc[~duplicated]
        warn('Weights has duplicated RSIDs, only the first, '
             'ocurrence will be kept')

    valid_alleles = ['A', 'C', 'G', 'T']
    if not np.isin(weights['ea'], valid_alleles).all():
        raise_error(
            'Effect allele in weights file are not valid. '
            f'Only the following values are possible: {valid_alleles}')

    return weights


def _find_chromosome_data(chr, datadir, subdir='imputation',
                          datalad_source='ria+http://ukb.ds.inm7.de#~genetic'):
    """Get a particular chromosome's (imputed) data from a datalad dataset or
    a directory

    Parameters
    ----------
    chr : str
        Chormosome to be fetched
    datadir : str or Path
        Directory to install the datalad dataset or search for files
    subdir : str, optional
        Directory in which the chromosome files are stored.
        Defaults to 'imputation'.
    datalad_source : str or None
        URI of the datalad dataset. If None, the chromosome data will be
        searched in the filesystem.
        Defaults to 'ria+http://ukb.ds.inm7.de#~genetic' (to be used at
        the INM-7 in FZJ)

    Returns
    -------
    files : list of str
        path to file names
    ds : datalad Dataset
        Datalad dataset object with chromome data, None if data is in a
        directory.
    got_files : list of files got using datalad

    """
    if not isinstance(datadir, Path):
        datadir = Path(datadir)

    ds = None
    if datalad_source:
        ds = datalad.install(  # type: ignore
            source=datalad_source, path=datadir)

    files = list((datadir / subdir).glob(f'*_c{chr}_*'))
    got_files = []
    if datalad_source:
        logger.info(f'Getting files: {files}')
        get_status = ds.get(files)  # type: ignore
        for f_val in get_status:
            status = f_val['status']
            if status != 'ok' and status != 'notneeded':
                ds.remove()  # type: ignore
                raise_error(
                    f'Error getting file {f_val["path"]} from datalad '
                    f'dataset: {status}')
            elif f_val['status'] == 'ok' and f_val['type'] == 'file':
                got_files.append(f_val['path'])
    if len(files) == 0:
        raise_error(f'No files were found on disk for chromosome {chr}')

    return files, ds, got_files


def genotype_from_datalad(
        rsids, workdir, datadir, chromosomes=None,
        datalad_source="ria+http://ukb.ds.inm7.de#~genetic",
        datalad_drop='get', recompute=False):
    """Reads a genotype from a datalad Dataset.
    A .bgen file with the selected RSIDs and chromosemes is created with
    qctool v2 (see notes) on the working directory.

    Parameters
    ----------
    rsids : str or list of str
        list of RSIDs to be included in the genotype
    workdir : str
        Path to save bgen and CSV files.
    datadir : str, optional
        Path to the datalad dataset or directory with the chromosome files.
    chromosomes : list of str | None
        Chromosomes to process. If None (default), use all chromosomes and get
        that information from ensmebl.org (requires internet connection)
        There should be one chromosome of each rsid provided.
    datalad_source : str, optional
        datalad data source, by default "ria+http://ukb.ds.inm7.de#~genetic"
    qctool : str, optional
        path to qctool, by default None
    datalad_drop : bool, optional
        If True, each chromosome file will be removed after the genotype is
        read. If False, the dataset will be kept. If 'get' (default), only the
        the files got during the process will be removed.
    recompute : bool, optional
        whether to recompute re-calculation (based on output file presence),
        by default False

    Returns
    -------
    Genotype
        The genotype

    Notes
    -----
    qctool must be installed (see https://www.well.ox.ac.uk/~gav/qctool/)

    """
    # check if qctool is available
    qctool = get_qctool()

    if not isinstance(workdir, Path):
        workdir = Path(workdir)

    if recompute is True and any(workdir.iterdir()):
        raise_error(
            f'To recompute, the working directory must be empty: {workdir}')
    workdir.mkdir(parents=True, exist_ok=True)

    df_chr = get_chromosomes(rsids, chromosomes=chromosomes)

    u_chr = df_chr['chromosomes'].unique()
    files = None
    files_to_read = []
    ds = None
    logger.info(f'Chromosomes needed: {u_chr}')
    for t_chr in u_chr:
        bgen_out = workdir / f'chromosome{t_chr}.bgen'
        rsid_out =  workdir / f'rsids_chromosome{t_chr}.txt'
        t_rsid = df_chr.loc[df_chr['chromosomes'] == t_chr, 'rsids'].values
        logger.info(f'Getting chromosome {t_chr} for RSID(s) {t_rsid}')

        if recompute is False and bgen_out.is_file():
            prev_rsid = pd.read_csv(
                rsid_out, sep='\t', header=None)  # type: ignore
            all_in = np.isin(t_rsid, prev_rsid.values)
            if all_in.all():
                logger.info(f'RSID(s) {t_rsid} already in {bgen_out}')
                files_to_read.append(bgen_out)
                continue
            else:
                raise_error(
                    f'Chromosome {t_chr} output file exists but does '
                    'not contain all the RSIDs required. '
                    'Set recompute=True to force generation of new files.')

        # get the data
        files, ds, got_files = _find_chromosome_data(
            t_chr, datadir, subdir='imputation', datalad_source=datalad_source)

        # find the bgen and sample files
        bgenfiles = [x for x in files if x.suffix == '.bgen']
        samplefiles = [x for x in files if x.suffix == '.sample']

        if len(bgenfiles) != 1 or len(samplefiles) != 1:
            raise_error(
                f'Wrong bgen and/or sample files for chromosome {t_chr}')
        bgen_in = bgenfiles[0]
        sample_in = samplefiles[0]

        # Convert to bgen with qctool
        df = pd.DataFrame(t_rsid)
        df.to_csv(rsid_out, index=False, header=False)

        cmd = (f'{qctool} -g {bgen_in} -s {sample_in} -incl-rsids {rsid_out} '
               f'-og {bgen_out} -ofiletype bgen_v1.2 -bgen-bits 8')

        logger.info(f'Converting to BGEN: {cmd}\n')
        result = subprocess.run(
            [cmd], shell=True, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise_error(f'Error converting to bgen: {result.stderr}')
        files_to_read.append(bgen_out)
        if datalad_drop is not False:
            if datalad_drop == 'get':
                for fname in got_files:
                    ds.drop(fname)  # type: ignore
            else:
                ds.drop(files)  # type: ignore

    return read_bgen(files_to_read)
