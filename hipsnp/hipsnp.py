import os
import shutil
import requests
import pandas as pd
import numpy as np
from datalad import api as datalad
from bgen_reader import open_bgen
from pathlib import Path
from hipsnp.utils import warn, raise_error, logger
import copy
from functools import reduce


def get_chromosome_data(c, datalad_source='ria+http://ukb.ds.inm7.de#~genetic',
                        imputation_dir='imputation', data_dir='/tmp/genetic'):
    """Get a particular chromosome's (imputed) data from a datalad dataset or
    a directoy 

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
        ds = datalad.clone(source=datalad_source, path=data_dir)
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
    """Make a REST call to ensemble.org and return a JSON object with
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

    url = 'http://rest.ensembl.org/variation/human/' + rsid +\
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
    for c, ch in enumerate(uchromosomes):

        file_out = Path(outdir, 'chromosome' + str(ch) + 'bgen')

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

        cmd = (qctool + ' -g ' + str(file_bgen) + ' -s ' + str(file_sample)
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
        warn(f'"weights" has duplicated RSIDs, only the first\
                appearane will be kept')

    if not np.isin(weights['ea'], ['A', 'C', 'T', 'G']).all():
        raise_error(f'effect allelel in weights is not "A", "C", "T", or "G"')

    return weights


class Genotype():
    """Genotype probabilities and metadata
    """
    def __init__(self, metadata, probabilities):
        """Data and metadata regarding genotype probabilites

        Paramters
        ---------
        metadata (Pandas Dataframe): metadata assoviated to a variant.
            indexes: RSIDS (str), columns: CHROM', 'POS', 'ID', 'FORMAT'
        probabilities (dict(str : tuple(str, 2d numpy array))):
                Dictionary where keys areRSIDS (str) and values are a tuple
                (Samples (str), probabilites (2d numpy array)).
                The first dimension of probabilites are Samples and
                the second dimension is 3 and contains probabilito of each
                allel combination REF-REF, ALT-REF, and ALT-ALT.
        """
        self._validate_arguments(metadata, probabilities)
        self._metadata = metadata  
        self._probabilities = probabilities 
        self._consolidated = False

    @property
    def rsids(self):
        """list of all rsids

        Returns
        -------
        list of str
            list containing the rsids presnt in the Genotype object
        """
        return list(self._metadata.index)

    @property
    def is_consolidated(self):
        """Wether the data in genotype is consoloidated. Consolidated data
        contains only rsids with samples which are present in all rsids.

        Returns
        -------
        bool
            True if consolidate
        """
        return self._consolidated

    @property
    def metadata(self):
        """[summary]

        Returns
        -------
        [type]
            [description]
        """        """"""
        """Metadata information with at least 'REF', 'ALT', 'CHROMosome', 
        'POSistion', 'ID', and 'FORMAT' as in .bgen files"""
        return self._metadata
    
    @property
    def probabilities(self):
        """ Probability of each allele combination foir each sample of an rsids

        Returns
        -------
        dict of tuples with a list of str and a 2d numpy array
            Dictionay keys are rsids. Dictionary values are a two elements
            tuple. The first element contains list with the sample names of the
            the probalities contained in the second element of the tuple. The
            second element of hte tuple is a numpy array with the first
            dimension equal to the length of the list of sample in the first
            elelment of the tuple. The second dimension of the numpy array has
            the three probalities associated to the allele combinations
            REF-REF, ALT-REF, and ALT-ALT. Therofe each row in the numpy array
            contains the three probabilites of each sample in the first element
            of the tuple.
        """
        return self._probabilities
    
    def unique_samples(self):
        """Return unique samples in Genotype.probability
        
        Returns
        -------
        list of strings
            unique samples in Genotype
        
        Rasises
        -------
        ValueError
            If Genotype RSIDs and Samples are not consolidated
        """
        if not self.is_consolidated:
            raise_error('Samples are not consolidated across RSIDs. Samples\
                must be first consolidated (see consolidatee() )')

        uniq_samples = self.probabilities[self.rsids[0]][0]
        return uniq_samples

    def filter(self, rsids=None, samples=None, weights=None, inplace=True):
        """Filter Genotype data object by Samples and rsids. rsids can be given
        explicitly to <rsids> and to the rsids in a file with weiths. The
        Genotype object will filter the common rsids from <rsids> and <weights> 

        Parameters
        ----------
        rsids : str or list of str | None
            rsids to keep. If None, does not filter (default).
        samples : str or list of str | None
            Samples to keep. If None, does not filter (default).
        weights : str or DataFrame, default None
            path to the csv file with the weigths or pandas DataFrame with
            weights as provided by read_weigts()
        inplace: bool
            If True retruns the same object, otherwise returns a new object

        Returns
        -------
        Genotype or None:
            Consolidated genotype probalities and metadata if inplace = False,
            None if inplace = True

        Rasises
        -------
        ValueError
            If the filtered data is empty.

        Notes
        -----
        If both filters are None, this method returns the same object, not
        a copy of it.
        """
        if rsids is None and samples is None and weights is None:
            warn(f'No paramters to filter were given to filter()')
            if inplace:
                return None
            else:
                return self
        
        if isinstance(weights, str):
            weights = read_weights(weights)
        
        if weights is not None:
            rsids_weights = weights.index.to_list()
            if rsids is None:
                rsids = rsids_weights
            else:
                if isinstance(rsids, str):
                    rsids = [rsids]
                rsids = list(np.intersect1d(rsids, rsids_weights))

        if inplace:
            self._filter_by_rsids(rsids=rsids, inplace=inplace)
            self._filter_by_samples(samples=samples, inplace=inplace)
            return None
        else:
            out = self._filter_by_rsids(
                rsids=rsids, inplace=inplace)._filter_by_samples(
                    samples=samples, inplace=inplace)
            return out

    def _filter_by_samples(self, samples=None, inplace=True):
        """Filter Genotype data object by Samples

        Parameters
        ----------
        samples : str or list of str | None
            Samples to keep. If None, does not filter (default).
        inplace: bool
            If True retruns the same object, otherwise returns a new object

        Returns
        -------
        Genotype or None:
            Consolidated genotype probalities and metadata if inplace = False,
            None if inplace = True

        Rasises
        -------
        ValueError
            If the filtered data is empty.

        Notes
        -----
        If the None, this method returns the same object, not
        a copy of it.
        """
        if samples is None:
            return self

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
            return None
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
            If True retruns the same object, otherwise returns a new object

        Returns
        -------
        Genotype or None:
            Filtred genotype probalities and metadata if inplace = False,
            None if inplace = True
            filtered genotype

        Rasises
        -------
        ValueError
            If the filtered data is empty.

        Notes
        -----
        If the None, this method returns the same object, not
        a copy of it.
        """
        if rsids is None:
            return self

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
        else:
            out = Genotype(metadata=meta_filtered,
                           probabilities=probs_filtered)
            return out

    def consolidate(self, inplace=True):
        """Align samples consistently across all RSIDs. If a sample is not found 
        in all RSID, the sample is discarded.

        Arguments

        inplace: bool
            If True retruns the same object, otherwise returns a new object

        Returns
        -------
        Genotype or None:
            Consolidated genotype probalities and metadata if inplace = False,
            None if inplace = True

        
        Rasises
        -------
        ValueError
            If the samples cannot be read or duplicated.
        """
        out = self._consolidate_samples(inplace)
        return out

    def _consolidate_samples(self, inplace):
        """Search for intersection and reorder"""
        # find common samples across all RSIDs
        common_samples = reduce(np.intersect1d,
                                (sp[0] for sp in self.probabilities.values()))
        if len(common_samples) == 0:
            raise_error('There are no samples common across all RSIDs')

        consol_prob_dict = {}
        for rsid, sample_prob in self.probabilities.items():
            # Get index of common_samples appearing on other RSIDs
            _, _, consol_idx = np.intersect1d(common_samples,
                                              sample_prob[0],
                                              assume_unique=True,
                                              return_indices=True)
            consol_prob_dict[rsid] = (common_samples,
                                      sample_prob[1][consol_idx, :])

        if inplace:
            self._probabilities = consol_prob_dict
            self._consolidated = True
            return None
        else:
            out = Genotype(metadata=self.metadata,
                           probabilities=consol_prob_dict)
            out._consolidated = True
            return out

    def get_array_of_probabilities(self):
        """Return a 3D array with the probabilties of the three allele
        combinations for all RSIDs and samples. To obtain the probalities, the 
        Genotype object data must be consolidate. (see consolidate())

        Retruns
        -------
        3D numpy array
            first dimension are rsid sorted as Genotype.rsids(). Second 
            dimension are samples sorted as Genotype.unique_samples(). Third 
            dimension are the probalities of the allele combinations
            REF-REF, ALT-REF, and ALT-ALT
        """
        if not self.is_consolidated:
            raise_error('Samples are not consolidated across RSIDs. Samples\
                must be consolidated (see consolidatee() )')

        prob_matrix = self._get_array_of_probabilites()
        return prob_matrix

    def _get_array_of_probabilites(self):
        """Iterate RSIDS over consolidated dictionary of probabilites to build
        a single 3d numpy array with all the probabiites."""
        n_samples = list(self.probabilities.values())[0][0].shape[0]
        consol_prob_matrix = np.zeros((len(self.probabilities),  # RSIDs
                                       n_samples,                # Samples
                                       3))                       # probability

        for i, smpl_prob in enumerate(self.probabilities.values()):
            consol_prob_matrix[i, :, :] = smpl_prob[1]
        return consol_prob_matrix

    def alleles(self, rsids=None, samples=None):
        """Get alleles for a Genotype and if weiths are given it computes the 
        risck scores.

        Parameters
        ----------
        rsids : list of str, optional
            rsids to be used, by default None
        samples : list of str, optional
            Samples to be used, by default None

        Returns
        -------
        genotype_allele: pandas DataFrame
            Alleles of each rsid and sample 
        
        genotype_012 : pandas DataFrame
            index of most probable allele 0=REF-REF , 1=ALTREF, 2=ALTALT 

        """
        if rsids is not None or samples is not None:
            gen_filt = self.filter(samples=samples, rsids=rsids, inplace=False)
        else:
            gen_filt = copy.deepcopy(self)
            
        if not gen_filt.is_consolidated:
            gen_filt.consolidate(inplace=True)

        probs = gen_filt.get_array_of_probabilities() 

        n_rsid = len(gen_filt.rsids)
        n_sample = len(gen_filt.unique_samples())

        logger.info(f'Calculating genotypes for {n_rsid} SNPs and \
                    {n_sample} samples ... ')

        genotype_allele = np.empty((n_rsid, n_sample), dtype=object)
        genotype_012 = np.zeros((n_rsid, n_sample), dtype=int)

        # resahpe to allow for straight indexing
        ref = np.tile(gen_filt.metadata['REF'].to_numpy(), (n_sample, 1)).T
        alt = np.tile(gen_filt.metadata['ALT'].to_numpy(), (n_sample, 1)).T

        i_max_p = np.argmax(probs, axis=2)
        genotype_allele[i_max_p == 0] = ref[i_max_p == 0] + ref[i_max_p == 0]
        
        # Sort needs a single array, but to add characters it needs two arrays 
        tmp = np.split(np.sort(np.vstack((ref[i_max_p == 1],
                                          alt[i_max_p == 1])).astype(str),
                               axis=1),
                       2, axis=0)
        g_allele = np.squeeze(np.char.add(tmp[0], tmp[1]))
        genotype_allele[i_max_p == 1] = g_allele

        genotype_allele[i_max_p == 2] = alt[i_max_p == 2] + alt[i_max_p == 2]
        genotype_012 = i_max_p

        genotype_allele = pd.DataFrame(data=genotype_allele,
                                       index=gen_filt.rsids,
                                       columns=gen_filt.unique_samples())
        genotype_012 = pd.DataFrame(data=genotype_012,
                                    index=gen_filt.rsids,
                                    columns=gen_filt.unique_samples())

        return genotype_allele, genotype_012

    def riskscore(self, weights, rsids=None, samples=None):
        """ Obtain risk score and dosage from Genotype object

        Parameters
        ----------
        weights : str or padas DataFrame,
            Path to CSV file with weights or pandas DataFrame with
            weights as provided by read_weigts()
        rsids : list of str, optional
            rsids to be used, by default None
        samples : list of str, optional
            Samples to be used, by default None


        Returns
        -------
        dosage : pandas DataFrame
            Dataframe with the dosage by rsid and samples
        riskscore : pandas DataFrame
            DataFrame with risksocre by samples
        """

        weights = read_weights(weights)

        gen_filt = self.filter(samples=samples,
                               rsids=rsids,
                               weights=weights,
                               inplace=False)

        if not gen_filt.is_consolidated:
            gen_filt.consolidate(inplace=True)

        # sort all DataFrames by the RSIDS in gen_filt.probabilities
        rsids_as_in_prob = list(gen_filt.probabilities.keys())
        gen_filt._metadata = gen_filt._metadata.reindex(rsids_as_in_prob)
        weights = weights.reindex(rsids_as_in_prob)
        
        n_rsid = len(gen_filt.rsids)
        n_sample = len(gen_filt.unique_samples())

        logger.info(f'Calculating riskscore for {n_rsid} SNPs and \
                    {n_sample} samples ... ')

        ref = np.tile(gen_filt.metadata['REF'].to_numpy(), (n_sample, 1)).T
        alt = np.tile(gen_filt.metadata['ALT'].to_numpy(), (n_sample, 1)).T
        probs = gen_filt.get_array_of_probabilities()

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
        riskscore = np.sum(dosage * wSNP, axis=0)

        dosage = pd.DataFrame(data=dosage,
                              columns=gen_filt.unique_samples(),
                              index=gen_filt.rsids)
        riskscore = pd.DataFrame(data=riskscore,
                                 index=gen_filt.unique_samples())
        return dosage, riskscore

    @classmethod
    def from_bgen(cls, files, verify_integrity=False):
        """Read bgen file. Return Genotype object with metadata and probabilites

        Parameters
        ----------
        files (list(str)): list with paths to bgen files.
        
        verify_integrity (bool, optional): Check for duplicates. 
            See pandas.concat(). Defaults to False.

        Returns
        -------
        Genotype object
        """
        metadata, probabilities =\
            cls._read_bgen_for_Genotype(files=files,
                                        verify_integrity=verify_integrity)

        return Genotype(metadata, probabilities)

    @staticmethod
    def _read_bgen_for_Genotype(files, verify_integrity=False):

        """Read bgen files and extract metadata and probabilites

        Parameters
        ----------
        files : str or list(str)
                Files to be read
        verify_integrity : bool, default False
                See pandas.concat documentation.

        Returns
        -------
            snpdata : pandas DataFrame
            row indexes: RSIDs; columns: CHROM, POS, ID, and FORMAT
            probabilites : dict
            keys: RDIDs, values: tuple(Smaples,
                                    probilities : numpy array [Samples, 3])
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

        snpdata = pd.DataFrame()
        probabilites = dict()
        for f in files:
            logger.info(f'Reading {f}')
            with open_bgen(f, verbose=False) as bgen:
                # we can only deal with biallelic variants
                if np.any(bgen.nalleles != 2):
                    raise_error('Only biallelic variants are allowed')

                # find duplicate RSIDs within a file
                _, iX_unique_in_file = np.unique(bgen.rsids, return_index=True)
                if iX_unique_in_file.shape[0] != bgen.rsids.shape[0]:
                    warn(f'Duplicated RSIDs in file {f}')

                # find duplicates with previous files
                if (not snpdata.empty and
                        np.sum(snpdata.index == bgen.rsids) != 0):
                    warn(f'Files have duplicated RSIDs')
                    # indexes with rsids not previously taken
                    # to keep unique RSIDS
                    mask_unique_btwb_files = np.isin(bgen.rsids, snpdata.index,
                                                     invert=True)
                    mask_to_keep = np.zeros(len(bgen.rsids), dtype=np.bool_)
                    mask_to_keep[iX_unique_in_file
                                 [mask_unique_btwb_files]] = True
                else:
                    mask_to_keep = np.ones(len(bgen.rsids), dtype=np.bool_)
        
                if any(mask_to_keep):

                    alleles = np.array(
                        [np.char.upper(val) for val in
                         np.char.split(bgen.allele_ids[mask_to_keep],
                                       sep=',')])
                        
                    if not np.isin(alleles, ['A', 'C', 'T', 'G']).all():
                        raise_error(f'alleles not "A", "C", "T", or "G"\
                                    in file {f}')

                    # dataframe with metadata of unique RSIDS.
                    tmp = pd.DataFrame(index=bgen.rsids[mask_to_keep])
                    tmp = tmp.assign(REF=alleles[:, 0],
                                     ALT=alleles[:, 1],
                                     CHROM=bgen.chromosomes[mask_to_keep],
                                     POS=bgen.positions[mask_to_keep],
                                     ID=bgen.ids[mask_to_keep],
                                     FORMAT='GP')

                    if f == files[0]:
                        myjoin = 'outer'
                    else:
                        myjoin = 'inner'
                    # concatenate metadata of files
                    snpdata = pd.concat([snpdata, tmp], join=myjoin, axis=0,
                                        verify_integrity=verify_integrity)

                    # crear probabilites data dictionary
                    probs = bgen.read()
                    tmp_probabilites = {k_rsid:
                                        (np.array(bgen.samples),
                                         np.squeeze(probs[:, i, :]))
                                        for i, k_rsid in
                                        enumerate(tmp.index)}
                    probabilites.update(tmp_probabilites)

                    tmp = None

        return snpdata, probabilites

    @staticmethod
    def _validate_arguments(meta, prob):
        """Basic check of Genotype arguments
            - metadata has columns 'REF', 'ALT', 'CHROM', 'POS', 'ID', 'FORMAT'
            - same order of RSID in metadata as probabilities, 
            - probabilities has same dimensions
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
