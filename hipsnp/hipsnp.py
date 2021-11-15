import io
from logging import log, warning
import os
import glob
import shutil
import requests
import pandas as pd
import numpy as np
from datalad import api as datalad
# from alive_progress import alive_it
import alive_progress
from bgen_reader import open_bgen
from pathlib import Path
from hipsnp.utils import warn, raise_error, logger
import copy
from functools import reduce


def get_chromosome(c,
                   datalad_source=None,
                   imputationdir='imputation',
                   data_dir=None):
    """
    get a particular chromosome's (imputed) data
    c: chromosome number, string
    source: datalad source, string (default: None which maps to
    'ria+http://ukb.ds.inm7.de#~genetic')
    imputationdir: directory in which the imputation files are stored, string
    (default: 'imputation')
    data_dir: directory to use for the datalad dataset, string
    (default: None which maps to '/tmp/genetic')
    returns: list of files, datalad dataset object, list of datalad get output
    """
    if datalad_source is None or datalad_source == '':
        files = glob.glob(os.path.join(ds.path,
                                       imputationdir,
                                       '*_c' + str(c) + '_*'))
        ds = None
        getout = ['datalad not used'] * len(files)
    else:
        if data_dir is None or data_dir == '':
            data_dir = os.path.join('/tmp', 'genetic')

        ds = datalad.clone(source=datalad_source, path=data_dir)
        files = glob.glob(os.path.join(ds.path,
                                       imputationdir,
                                       '*_c' + str(c) + '_*'))
        getout = ds.get(files)

    return files, ds, getout


def ensembl_human_rsid(rsid):
    """
    make a REST call to ensemble and return json info of a variant given a rsid
    rsid: string
    returns: json object
    """
    if not isinstance(rsid, str) or rsid[0:2] != 'rs':
        print(rsid + '\n')
        print('rsid must be a string with a starting "rs"')
        raise ValueError

    url = 'http://rest.ensembl.org/variation/human/' + rsid +\
          '?content-type=application/json'
    response = requests.get(url)
    return response


def rsid2chromosome(rsids, chromosomes=None):
    """
    get the chromosome of each rsid
    rsids: list of rsids, string or list of strings
    chromosomes: list of chromosomes, string or list of strings
    returns: dataframe with columns 'rsids' and 'chromosomes'
    """
    if isinstance(rsids, str) and os.path.isfile(rsids):
        rsids = pd.read_csv(rsids, header=None, sep='\t', comment='#')
        if rsids.shape[1] > 1:
            # this check provides support for PSG files
            if isinstance(rsids.iloc[0, 1], str):
                rsids.drop(index=0, inplace=True)
            chromosomes = list(rsids.iloc[:, 1])
            chromosomes = [str(c) for c in chromosomes]
        rsids = list(rsids.iloc[:, 0])
    elif isinstance(rsids, str):
        rsids = [rsids]

    if chromosomes is None:
        # get from ensembl
        chromosomes = [None] * len(rsids)
        for rs in range(len(rsids)):
            ens = ensembl_human_rsid(rsids[rs])
            ens = ens.json()
            ens = ens['mappings']
            for m in range(len(ens)):
                if ens[m]['ancestral_allele'] is not None:
                    chromosomes[rs] = ens[m]['seq_region_name']
    else:
        assert len(chromosomes) == len(rsids)
        if isinstance(chromosomes, str) or isinstance(chromosomes, int):
            chromosomes = [chromosomes]
        chromosomes = [str(c) for c in chromosomes]

    df = pd.DataFrame()
    df['chromosomes'] = chromosomes
    df['rsids'] = rsids
    return df


def rsid2snp(rsids, 
             outdir,
             datalad_source="ria+http://ukb.ds.inm7.de#~genetic",
             qctool=None,
             datalad_drop=True,
             datalad_drop_if_got=True,
             data_dir=None,
             force=False,
             chromosomes=None,
             chromosomes_use=None,
             outformat='bgen'):
    """
    convert rsids to snps
    rsids: list of rsids or a file with rsids, string or list of strings
    datalad_source: datalad source, string
    (default: 'ria+http://ukb.ds.inm7.de#~genetic')
    qctool: path to qctool, string (default: None, which maps to 'qctool')
    datalad_drop: whether to drop the datalad dataset after getting the files,
    bool (default: True)
    datalad_drop_if_got: whether to drop files only if downloaded with get,
    bool (default: True)
    data_dir: directory to use for the (datalad) dataset, string (default:
    None which maps to '/tmp/genetic')
    force: whether to force re-calculation (based on output file presence),
    bool (default: False)
    chromosomes: list of chromosomes to process, list of strings
    (default: None which uses all chromosomes)
    ch_rs: dataframe with columns 'rsids' and 'chromosomes', dataframe
     (default: None)
    returns: a pandas dataframe with rsid-chromosome pairs and the vcf files
    are created in the outdir
    outformat: output file format 'bgen' or 'vcf', string (default: 'bgen')
    """
    assert isinstance(outformat, str) and outformat in ['vcf', 'bgen']
    # check if qctool is available
    if qctool is None:
        qctool = shutil.which('qctool')

    if qctool is None or os.path.isfile(qctool) is False:
        print('qctool is not available')
        raise

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if force is True and os.listdir(outdir):
        print('the output directory must be empty')
        raise

    # get chromosome of each rsid
    if chromosomes is not None:
        assert len(chromosomes) == len(rsids)

    ch_rs = rsid2chromosome(rsids, chromosomes=chromosomes)
    chromosomes = ch_rs['chromosomes'].tolist()
    uchromosomes = pd.unique(chromosomes)
    files = None
    ds = None
    print('chromosomes needed: ' + str(uchromosomes) + '\n')
    for c in range(len(uchromosomes)):
        ch = uchromosomes[c]
        if chromosomes_use is not None and ch not in chromosomes_use:
            print('skipping chromosome ' + str(ch), ' not in the use list')
            continue
        file_out = os.path.join(outdir,
                                'chromosome_' + str(ch) + '.' + outformat)
        if force is False and os.path.isfile(file_out):
            print(f'chromosome {ch} output file exists, skipping: {file_out}')
            continue

        ind = [i for i, x in enumerate(ch_rs['chromosomes'])
               if x == uchromosomes[c]]
        rs_ch = [rsids[i] for i in ind]
        print(f'chromosome {ch} with {len(rs_ch)} rsids')
        if len(rs_ch) == 0:
            continue

        if len(rs_ch) < 11:
            print('rsids: ' + str(rs_ch) + '\n')

        # get the data
        print('datalad: getting files')
        files, ds, getout = get_chromosome(ch,
                                           datalad_source=datalad_source,
                                           data_dir=data_dir)
        for fi in range(len(getout)):
            status = getout[fi]['status']
            print('datalad: status ' + str(status) + ' file ' + str(files[fi]))
            if status != 'ok' and status != 'notneeded':
                print('datalad: error getting file '
                      + str(fi) + ': ' + str(getout[fi]['path']) + '\n')
                # todo: cleanup datalad files
                raise

        # find the bgen and sample files
        file_bgen = None
        file_sample = None
        for fl in files:
            name, ext = os.path.splitext(fl)
            if ext == '.bgen':
                assert file_bgen is None
                file_bgen = fl
            elif ext == '.sample':
                assert file_sample is None
                file_sample = fl

        assert file_bgen is not None and file_sample is not None
        file_rsids = os.path.join(outdir,
                                  'rsids_chromosome' + str(ch) + '.txt')
        df = pd.DataFrame(rs_ch)
        df.to_csv(file_rsids, index=False, header=False)

        cmd = (qctool + ' -g ' + file_bgen + ' -s ' + file_sample
               + ' -incl-rsids ' + file_rsids  + ' -og ' + file_out)
        if outformat == 'bgen':
            cmd += ' -ofiletype bgen_v1.2 -bgen-bits 8'

        print('running qctool: ' + cmd  + '\n')
        os.system(cmd)

        if datalad_drop:
            # must use relative paths???
            common_prefix = os.path.commonprefix([files[0], ds.path])
            files_rel = [os.path.relpath(path, common_prefix)
                         for path in files]
            if datalad_drop_if_got:
                for fi in range(len(getout)):
                    if getout[fi]['status'] == 'ok' and getout[fi]['type'] == 'file':
                        print(f'datalad: dropping file {files_rel[fi]}')
                        ds.drop(files_rel[fi])
            else:
                print('datalad: dropping all files\n')
                ds.drop(files_rel)

        print('done with chromosome ' + str(ch) + '\n')

    return ch_rs, files, ds


def rsid2snp_multiple(files, outdir,
                      qctool=None,
                      datalad_source="ria+http://ukb.ds.inm7.de#~genetic",
                      data_dir=None,
                      datalad_drop=True,
                      outformat='bgen'):
    """
    """
    chromosomes = []
    # check if all files are available
    outdirs = [None] * len(files)
    ch_rs   = [None] * len(files)
    for i in range(len(files)):
        print('file ' + str(i) + ': ' + str(files[i]))
        if os.path.isfile(files[i]) is False:
            print('file ' + str(files[i]) + ' does not exist')
            raise
        bname = os.path.basename(files[i])
        bname = os.path.splitext(bname)[0]
        outdirs[i] = os.path.join(outdir, bname)
        ch_rs[i] = rsid2chromosome(files[i])
        print(ch_rs[i].head())
        uchromosomes = pd.unique(ch_rs[i]['chromosomes'])
        print(uchromosomes)
        chromosomes = chromosomes + uchromosomes.tolist()

    chromosomes = pd.unique(chromosomes)
    print('chromosomes: ' + str(chromosomes))
    print('#files: ' + str(len(files)) + '\n')
    for c in range(len(chromosomes)):
        ch = chromosomes[c]            
        for i in range(len(files)):
            print(f'chromosome {ch} rsids {files[i]}  outdir {outdirs[i]}')
            datalad_drop_i = False
            if i == len(files) - 1:
                datalad_drop_i = datalad_drop
            chrs, chfiles, ds = rsid2snp(rsids=ch_rs[i]['rsids'].tolist(),
                                         outdir=outdirs[i],
                                         # chromosomes=
                                         # ch_rs[i]['chromosomes'].tolist(),
                                         # for testing, or the rsid2chromosome
                                         # cannot find mock data
                                         chromosomes='1',
                                         datalad_source=datalad_source,
                                         data_dir=data_dir,
                                         qctool=qctool,
                                         chromosomes_use=[ch],
                                         force=False,
                                         datalad_drop=datalad_drop_i,
                                         datalad_drop_if_got=False,
                                         outformat=outformat)

    return outdirs


def read_bgen(files,
              rsids_as_index=True,
              no_neg_samples=False,
              join='inner',
              verify_integrity=False,
              probs_in_pd=False,
              verbose=True):
    if isinstance(files, str):
        files = [files]

    # make sure that files exist
    for f in files:
        assert os.path.isfile(f)

    # read all the files
    if verbose:
        print('reading ' + str(len(files)) + ' bgen files... ')
    probsdata = dict()
    snpdata = pd.DataFrame()
    for f in alive_progress.alive_it(files):
        if verbose:
            print('reading ' + f)
        bgen = open_bgen(f, verbose=False)
        # sanity checks
        # we can only deal with biallelic variants
        nalleles = np.unique(bgen.nalleles)
        assert len(nalleles) == 1 and nalleles[0] == 2
        probs = bgen.read()  # read the probabilities [samples, snps?, 3]

        tmp = pd.DataFrame(index=range(len(bgen.rsids)))
        if rsids_as_index:
            tmp.index = bgen.rsids

        if probs_in_pd:
            # this is extremely slow
            print('putting probabilities in the dataframe')
            for j in range(probs.shape[1]):  # snps
                for i in range(probs.shape[0]):  # sample
                    tmp.iloc[j][i] = probs[i, j, :]
        else:
            nkey = len(probsdata.keys())
            probsdata[str(nkey)] = dict()
            probsdata[str(nkey)]['probs'] = np.array(probs)
            probsdata[str(nkey)]['samples'] = np.array(bgen.samples)
            probsdata[str(nkey)]['rsids'] = np.array(bgen.rsids)
        
        # add more data
        # get REF and ALT
        alleles = bgen.allele_ids
        alleles = np.array([a.split(',') for a in alleles])
        tmp = tmp.assign(REF=alleles[:, 0], ALT=alleles[:, 1])
        tmp = tmp.assign(CHROM=bgen.chromosomes,
                         POS=bgen.positions,
                         ID=bgen.ids)
        tmp['FORMAT'] = 'GP'

        myjoin = join
        if f == files[0]:
            myjoin = 'outer'

        snpdata = pd.concat([snpdata, tmp],
                            join=myjoin,
                            axis=0,
                            verify_integrity=verify_integrity)
        tmp = None

    return snpdata, probsdata


def read_bgen_for_Genotype(files, verify_integrity=False, verbose=True):

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
        raise_error('bgen file does not exist', FileNotFoundError())

    # read all the files
    logger.info(f'Reading {len(files)} bgen files...')

    snpdata = pd.DataFrame()
    probabilites = dict()
    for f in alive_progress.alive_it(files):
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
            if not snpdata.empty and np.sum(snpdata.index == bgen.rsids) != 0:
                warn(f'Files have duplicated RSIDs')
                # indexes with rsids not previously taken to keep unique RSIDS
                mask_unique_btwb_files = np.isin(bgen.rsids, snpdata.index,
                                                 invert=True)
                mask_to_keep = np.zeros(len(bgen.rsids), dtype=np.bool_)
                mask_to_keep[iX_unique_in_file[mask_unique_btwb_files]] = True
            else:
                mask_to_keep = np.ones(len(bgen.rsids), dtype=np.bool_)
      
            if any(mask_to_keep):
                # get REF and ALT
                alleles = bgen.allele_ids[mask_to_keep]
                alleles = np.array([a.split(',') for a in alleles])
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
                                    for i, k_rsid in enumerate(tmp.index) }
                probabilites.update(tmp_probabilites)

                tmp = None

    return snpdata, probabilites


def read_weights(weights):
    """
    read weights from a file
    """
    if isinstance(weights, str):
        assert os.path.isfile(weights)
        weights = pd.read_csv(weights, sep='\t', comment='#')
    weights.columns = [x.lower() for x in weights.columns]
    weights.rename(columns={'snpid': 'rsid', 'chr_name': 'chr',
                            'effect_allele': 'ea', 'effect_weight': 'weight'},
                   inplace=True)

    assert 'ea' in weights.columns
    assert 'weight' in weights.columns
    assert 'rsid' in weights.columns
    rsids = weights['rsid'].tolist()
    # make sure that all rsids are unique
    assert len(rsids) == len(set(rsids))
    return weights


def snp2genotype(snpdata, th=0.9, snps=None, samples=None,
                 genotype_format='allele', probs=None, weights=None,
                 verbose=True, profiler=None):
    """
    given a snp file path(s) or a pandas df from read_vcf/read_bgen
    returns genotypes and probabilities
    """
    if profiler is not None:        
        profiler.enable()

    assert isinstance(genotype_format, str)
    assert genotype_format in ['allele', '012']

    if not isinstance(snpdata, pd.DataFrame):
        print("don't know how to handle the input")
        print('please use read_vcf or read_bgen to get the required input')
        raise AttributeError

    nsnp = snpdata.shape[0]
    ncol = snpdata.shape[1]
    if samples is None:
        if verbose:
            print('no samples specified, using all')
        if probs is None:
            samples = [snpdata.columns[i] for i in range(9, ncol)]
        else:
            pkeys = list(probs.keys())  # number of bgen files read.
            samples = probs[pkeys[0]]['samples']
            for pk in range(1, len(pkeys)):
                # if there are several bgen files (one per RSID),
                samples = np.append(samples, probs[pkeys[pk]]['samples'])
                # the samples of each file are appened togetehr
                samples = np.unique(samples)
                # onlyt not repeated samples are kept and sorted
    else:
        if probs is None:
            assert all(sam in snpdata.columns for sam in samples)

    if snps is None:
        snps = list(snpdata.index)
    # else:
    #    assert all(snp in list(snp.index) for s in snps)

    riskscore = None
    if weights is not None:
        weights = read_weights(weights)
        if verbose:
            print('will calculate riskscore using weights')
            print(weights.head())
        riskscore = pd.DataFrame(0.0, columns=range(1), index=samples,
                                 dtype=float)

    genotype_allele = pd.DataFrame('', columns=snps, index=samples, dtype=str)
    genotype_012 = pd.DataFrame(np.nan, columns=snps, index=samples,
                                dtype=float)
    probability = genotype_012.copy()
    print('calculating genotypes for ' + str(len(snps)) + ' SNPs and ' +
          str(len(samples)) + ' samples ... ')
    for snp in alive_progress.alive_it(snps):
        # get SNP info
        try:
            REF = snpdata['REF'][snp]
            ALT = snpdata['ALT'][snp]
            assert len(REF) == 1 and len(ALT) == 1
            if weights is not None:
                EA  = weights['ea'][weights['rsid'] == snp].values
                if len(EA) == 0:
                    weightSNP = None
                else:
                    EA = EA[0]
                    assert isinstance(EA, str) and len(EA) == 1
                    weightSNP = weights['weight'][weights['rsid'] == snp].values[0]
                    weightSNP = float(weightSNP)
        except Exception as e:
            if verbose:
                print('error parsing snp ' + str(snp))
                print(e)
            genotype_allele.loc[snp, :] = np.nan
            genotype_012.loc[snp, :] = np.nan
            probability.loc[snp, :] = pd.Series([[np.nan] * 3] * len(samples))
            continue
        # get a df with probabilities for all the samples
        GP = None
        try:
            if probs is not None:
                for pk in probs.keys():
                    if snp in probs[pk]['rsids']:
                        snpidx = np.where(probs[pk]['rsids'] == snp)
                        assert len(snpidx) == 1
                        snpidx = snpidx[0][0]

                        index = probs[pk]['samples']  # samples in probs
                        index_mask = np.in1d(index, samples)
                        # boolean of elements in ar1 present in ar2
                        probs_pk = probs[pk]['probs'][:, snpidx, :]
                        probs_pk = probs_pk[index_mask]
                        # why a mask? all samples are taken always because
                        # the unique samples are kept
                        # probs is samples x snps x 3
                        GP = pd.DataFrame(probs_pk, index=index[index_mask],
                                          columns=range(3))
                        # keep what we need
                        # todo check if this reorders
                        # GP = GP.reindex(samples).dropna()
                        # break
            else:
                # todo: vectorize this
                GP = pd.DataFrame(index=samples, columns=range(3))
                for sam in samples:
                    try:
                        gt, gp = parse_GTGP(snpdata[sam][snp])
                        GP.loc[sam, :] = gp
                    except Exception as e:
                        if verbose:
                            print('error parsing snp ' + str(snp))
                            print(e)
                        genotype[sam][snp] = np.nan
                        probability[sam][snp] = [np.nan] * 3
                        continue
        except Exception as e:
            print(e)

        # use GP to calculate genotype
        imax = np.argmax(GP.values, axis=1)
        genotype_allele.loc[GP.index[imax == 0], snp] = REF + REF
        genotype_allele.loc[GP.index[imax == 1], snp] = "".join(sorted(REF
                                                                       + ALT))
        genotype_allele.loc[GP.index[imax == 2], snp] = ALT + ALT
        genotype_012.loc[GP.index, snp] = imax

        # todo fix this
        # probability.loc[snp, GP.index] = GP.values

        if weights is not None and weightSNP is not None:
            # todo this is quite slow and maybe incorrect
            # print('SNP ' + snp + ' EA ' + EA + ' REF ' + REF +\
            #      ' ALT ' + ALT + ' weight ' + str(weightSNP))
            dosage = GP2dosage(GP, REF, ALT, EA)
            # whichever samples have a non-nan dosage get added
            # all others become NaN which ius good
            riskscore = riskscore.add(weightSNP * dosage, axis=0)

    if profiler is not None:
        profiler.disable()

    if weights is None:
        return genotype_allele, genotype_012, probability
    else:
        return genotype_allele, genotype_012, riskscore


def GP2dosage(GP, REF, ALT, EA):
    assert GP.shape[1] == 3
    if EA == REF:
        dosage = GP.iloc[:, 1] + 2 * GP.iloc[:, 0]
    elif EA == ALT:
        dosage = GP.iloc[:, 1] + 2 * GP.iloc[:, 2]
    else:

        print('SNP ' + snp + ' ALT ' + ALT + ' or REF ' + REF +
              'do not match EA ' + EA)
        raise
    return dosage


def parse_GTGP(GTGP, format=None):
    """
    given a GT:GP string, returns GT and an array of three GP
    """
    assert isinstance(GTGP, str) or isinstance(GTGP, list) or isinstance(GTGP, np.ndarray)
    GT = None
    if isinstance(GTGP, list) or isinstance(GTGP, np.ndarray):
        GP = GTGP
    else:
        if format is None:
            format = ':' in GTGP
        if format:
            GTGP = GTGP.split(':')
            assert len(GTGP) == 2
            GT = GTGP[0]
            GP = GTGP[1]
        else:
            GP = GTGP

        try:
            GP = [float(gp) for gp in GP.split(',')]
            assert len(GP) == 3
        except Exception:
            GP = [np.nan, np.nan, np.nan]
    GP = np.array(GP)
    return GT, GP


class Genotype():
    """Genotype probabilities and metadata
    """
    def __init__(self, metadata, probabilities):
        """Data and metadata regarding genotype probabilites

        Args:
            metadata (Pandas Dataframe): metadata assoviated to a variant.
            indexes: RSIDS (str), columns: CHROM', 'POS', 'ID', 'FORMAT'
            probabilities (dict(str : tuple(str, 2d numpy array))):
                Dictionary where keys are
            RSIDS (str) and values are a tuple
                (Samples (str), probabilites (2d numpy array)).
            The first dimension of probabilites are Samples and
                the second dimension is 3.
        """
        self._validate_arguments(metadata, probabilities)
        self._metadata = metadata  # pandas dataframe == snp
        self._probabilities = probabilities  # dicttuple(samples, probabilites)
        self._consolidated = False

    @property
    def rsids(self):
        return list(self._metadata.index)

    @property
    def is_consolidated(self):
        return self._consolidated

    @property
    def metadata(self):
        return self._metadata
    
    @property
    def probabilities(self):
        return self._probabilities
    
    def unique_samples(self):
        """Return unique samples in Genotype.probability
        
        Returns
        -------
        list of strings
            unique samples in Genotype
        """
        pass

    def filter(self, rsids=None, samples=None, inplace=True):
        """Filter Genotype data object by Samples

        Parameters
        ----------
        samples : str or list of str | None
            Samples to keep. If None, does not filter (default).

        Returns
        -------
        out = Genotype
            filtered genotype.

        Rasises
        -------
        ValueError
            If the filtered data is empty.

        Notes
        -----
        If both filters are None, this method returns the same object, not
        a copy of it.
        """
        # TODO: make possible that filter happnes inplace
        if rsids is None and samples is None:
            return self
        else:
            out = self._filter_by_rsids(
                rsids=rsids)._filter_by_samples(samples=samples)
        return out

    def _filter_by_samples(self, samples=None):
        """Filter Genotype data object by Samples

        Parameters
        ----------
        samples : str or list of str | None
            Samples to keep. If None, does not filter (default).

        Returns
        -------
        out = Genotype
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

        out = Genotype(metadata=meta_filtered, probabilities=probs_filtered)
        return out

    def _filter_by_rsids(self, rsids=None):
        """Filter Genotype data object by RSID

        Parameters
        ----------
        samples : str or list of str | None
            RSIDs to keep. If None, does not filter (default).

        Returns
        -------
        out = Genotype
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
                          for k_rsid in rsids}

        out = Genotype(metadata=meta_filtered, probabilities=probs_filtered)
        return out

    def consolidate(self, inplace=True):
        """Align samples consistently across all RSIDs. If a sample is not found 
        in all RSID, the sample is discarded.

        Arguments

        inplace: bool
            If True retruns the same object, otherwise returns a new object

        Returns
        -------
        consol_gen = Genotype:
            Consolidated genotype probalities and metadata
        prob_matrix = numpy.ndarray
            3 dimensional array with RSIDS, samples, and probabilites on each 
            dimension respectively
        
        Rasises
        -------
        ValueError
            If the samples cannot be read or duplicated.
        """
        # TODO: DONE do not return 3d prob. matrix. To get the MAtrix should 
        # be another method
        out = self._consolidate_samples(inplace)
        return out

    def _consolidate_samples(self, inplace):
        """Search for intersection and reorder"""
        # find common samples across all RSIDs
        # FIXME: FAils on test 4. If one RSID has no common samples with other RSIDS
        # reduce(intersect1d) cannot find common samples across al RSIDS
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

    def _validate_samples_and_get_IDs(self):
        """"Get the ID number of each sample and check that there no duplicated
        samlpes within an RSID. Though an error there are duplicates samples or 
        the digits of a sample cannot converted to integers. Adds to atributes
        to Genotype: the IDs of the samples per RSID and number of samples on 
        each RSID"""
        # FIXME: move this validation to cration of Genotype object?
        self._samples_ID = {}
        self._n_samples = {}
        for key_rsid, val_rsid in self.probabilities.items():
            try:
                tmp_samplesID = [int(sample[7:]) for sample in val_rsid[0]]
            except ValueError:
                raise_error(f'Wrong Sample indentifier')
            
            if len(set(tmp_samplesID)) < len(val_rsid[0]):
                raise_error(f'There are duplicated samples in {key_rsid}')
            
            self._samples_ID[key_rsid] = np.array(tmp_samplesID)
            self._n_samples[key_rsid] = len(tmp_samplesID)

    def _consolidate_samples_intersection(self):
        """Samples and associated probalities of all RSIDs are reordered to
        be consistent acrross all RSIDs. The RSID with the least number of
        samples is used as prdering references for all other RSIDS, samples
        inconsistent across all other RSIDs are discarded. Returns a Genotype
        object and 3D numpy array with all problities """
        # use RSID with the least number of samples to rearange all other RSIDs
        
        # get RSID with the least number of samples
        least_samples_rsid = min(self._n_samples, key=self._n_samples.get)

        consol_prob_dict = {}
        ref_samples_ID = self._samples_ID[least_samples_rsid]
        ref_samples_str = self.probabilities[least_samples_rsid][0]

        consol_prob_matrix = np.zeros((len(self.probabilities),  # RSIDs
                                       ref_samples_ID.shape[0],   # Samples
                                       3))                        # 3 probabil.
        idx_not_consolidated  = []
        rsid_not_consolidated = []
        for i, dict_prob in enumerate(self.probabilities.items()):
            _, _, consol_idx = np.intersect1d(ref_samples_ID,
                                              self._samples_ID[dict_prob[0]],
                                              assume_unique=True,
                                              return_indices=True)
            if consol_idx.shape[0] > 0:  # sucessful consolidation
                consol_prob_dict[dict_prob[0]] = (ref_samples_str,
                                                  dict_prob[1][1]
                                                  [consol_idx, :])
                consol_prob_matrix[i, :, :] =\
                    consol_prob_dict[dict_prob[0]][1]
            else:
                warn(f'RSID {dict_prob[0]} cannot be consolidated as it has\
                    not matching samples with other RSIDs')
                idx_not_consolidated.append(i)
                rsid_not_consolidated.append(dict_prob[0])

        # Adjust metadata and matrix of probabilites for not consolidated RSIDs
        if len(idx_not_consolidated) > 0:
            consol_prob_matrix = np.delete(consol_prob_matrix,
                                           idx_not_consolidated, axis=0)
            meta_cosolidated = self._metadata.drop(index=rsid_not_consolidated)
            out_Genotype = Genotype(metadata=meta_cosolidated,
                                    probabilities=consol_prob_dict)
            out_Genotype._consolidated = True
        else:
            out_Genotype = Genotype(metadata=self._metadata,
                                    probabilities=consol_prob_dict)
            out_Genotype._consolidated = True
            out_Genotype._unique_samples = ref_samples_str
        return consol_prob_matrix, out_Genotype

    def get_array_of_probabilites(self):
        """Return a 3D array with the probabilties of all RSIDs and samples. If 
        Genotype is not consolidated, it is first consolidated
        """
        if not self.is_consolidated:
            raise_error('Samples are not consolidated across RSIDs. Samples\
                must be consolidated (see consolidatee() )')

        prob_matrix = self._get_array_of_probabilites()
        return prob_matrix

    def _get_array_of_probabilites(self):
        """Oterate RSIDS over consolidated dictionary of probabilites to build a
        single 3d numpy array with all the probabiites."""
        # TODO: DONE returns 3d matrix of probabilites
        n_samples = list(self.probabilities.values())[0][0].shape[0]
        consol_prob_matrix = np.zeros((len(self.probabilities),  # RSIDs
                                       n_samples,                # Samples
                                       3))                       # probability
        for i, sample_prob in enumerate(self.probabilities.values()):
            consol_prob_matrix[i, :, :] = sample_prob[1]

        return consol_prob_matrix

    def validate_metadata(self):
        """ Check that metadata information has the right format. Return the 
        same Genotype if the filds REF and ALT contain only one lement. 
        Otherwise, remove the RSIDS with worng metadata in the metadata and 
        probability fields and retrun a new instance of Genotype
        """
        # TODO: inplace option
        wrong_rsids = []
        for rsid, ref, alt in zip(self.metadata.index,
                                  self.metadata['REF'].values,
                                  self.metadata['ALT'].values):
            
            if len(ref) != 1 or len(alt) != 1 or\
               not isinstance(ref, str) or not isinstance(alt, str):
                wrong_rsids.append(rsid)
        
        if len(wrong_rsids) > 0:
            metadata = self.metadata.drop(index=wrong_rsids)
            prob = copy.deepcopy(self.probabilities)
            for rsid in wrong_rsids:
                prob.pop(rsid)
            return Genotype(metadata=metadata, probabilities=prob)
        else:
            return self
        # TODO: pytest it
        # FIXME: Should it return a new object or modiffies the existing object

    def read_weights(self, weights):
        """read weights from a file
        """
        # TODO: not part of Genotype. Only one function with validation
        try:
            weights = pd.read_csv(weights, sep='\t', comment='#')
            weights.columns = [x.lower() for x in weights.columns]
            # FIXME: shouldn't be rsids the index to be consistent with metadata
            weights.rename(columns={'snpid': 'rsid', 'chr_name': 'chr',
                                    'effect_allele': 'ea',
                                    'effect_weight': 'weight'},
                           inplace=True)
        except ValueError as e:
            raise_error(f'Fails reading weights', klass=e)
        
        weights = self._validate_weights(weights)
        return weights

    def _validate_weights(weights):
        """check that the weights DataFrame has the right format
        """

        if not('ea' and 'weight' and 'rsid' in weights.columns):
            raise_error(f'Weights contains wrong information')

        rsids = weights['rsid'].tolist()
        if len(rsids) == len(set(rsids)):
            warning(f'"weights" has duplicated RSIDs, only the first\
                      appearane is kept')
            
            # FIXME: AQUI ESTOY. Esta mal. rsids es una columna no el index.
            dopple_rsids = rsids[weights.duplicated()]
            weights.drop(index=dopple_rsids)

        assert 'ea' in weights.columns
        assert 'weight' in weights.columns
        assert 'rsid' in weights.columns
        rsids = weights['rsid'].tolist()
        # make sure that all rsids are unique
        assert len(rsids) == len(set(rsids))
        return weights


    # if isinstance(weights, str):
    #     assert os.path.isfile(weights)
    #     weights = pd.read_csv(weights, sep='\t', comment='#')
    # weights.columns = [x.lower() for x in weights.columns]
    # weights.rename(columns={'snpid': 'rsid', 'chr_name': 'chr',
    #                         'effect_allele': 'ea', 'effect_weight': 'weight'},
    #                inplace=True)

    # assert 'ea' in weights.columns
    # assert 'weight' in weights.columns
    # assert 'rsid' in weights.columns
    # rsids = weights['rsid'].tolist()
    # # make sure that all rsids are unique
    # assert len(rsids) == len(set(rsids))
    # return weights

    @staticmethod
    def from_bgen_transform(files, rsids_as_index=True, no_neg_samples=False, \
            join='inner', verify_integrity=False, \
            probs_in_pd=False, verbose=True):
        """Read bgen files and return Genotype Obj. It wraps read_bgen as origianaly
        written and transforms output to fit Genotype Obj.

        Args:
            files (str): bgen files
            rsids_as_index (bool, optional): [description]. Defaults to True.
            no_neg_samples (bool, optional): [description]. Defaults to False.
            verify_integrity (bool, optional): [description]. Defaults to False.
            verbose (bool, optional): [description]. Defaults to True.

        Returns:
            Genotype: Object with metadata and probabilities
        """  

        if isinstance(files, list):
            assert len(files) == len(set(files)),\
                "There are duplicate bgen files"
      
        metadata, probs = read_bgen(files, rsids_as_index=True,
                                    no_neg_samples=False,
                                    join='inner', verify_integrity=False,
                                    probs_in_pd=False, verbose=True)

        probabilities = dict()
        for prob_key in probs:  # iterate bgen files saved to a dict
            for i, rsid in enumerate(probs[prob_key]['rsids']):
                if rsid not in probabilities.keys():
                    probabilities[rsid] = (probs[prob_key]['samples'], 
                                           np.squeeze(probs[prob_key]['probs'][:, i, :]))
        return Genotype(metadata, probabilities)

    @staticmethod
    def from_bgen(files, verify_integrity=False, verbose=True):
        """Read bgen data and return Genotype object with metadata and probabilites

        Args:
            files (list(str)): list with paths to bgen files.
            verify_integrity (bool, optional): Check for duplicates. 
            See pandas.concat(). Defaults to False.
            verbose (bool, optional): addiotinal processing information. 
            Defaults to True.
        """                                             
        metadata, probabilities = read_bgen_for_Genotype(files=files, 
                                                       verify_integrity=verify_integrity,
                                                       verbose=verbose)


        return Genotype(metadata, probabilities) 

    @staticmethod
    def _validate_arguments(meta, prob):
        """check Genotype arguments: 
        - metadata DataFrame has columns 'REF', 'ALT', 'CHROM', 'POS', 'ID', 'FORMAT'
        - same order of RSID in metadata as probabilities, 
        - probabilities has same dimensions"""

        if sum((col in ['REF', 'ALT', 'CHROM', 'POS', 'ID', 'FORMAT'] for col in meta.columns)) < 6:
            raise_error("Missign columns in metadata")
        if sorted(meta.index) != sorted(prob.keys()):
            raise_error("Mismatch of RSIDs between metadata and probabilities")
        if any([len(prob[k_key][0]) != prob[k_key][1].shape[0] or
                prob[k_key][1].shape[1] != 3 for k_key in prob.keys()]):
            raise_error("Mismatch dimensions between samples and probabilities")

def snp_Genotype_2genotype(snpdata, gen, th=0.9, snps=None, samples=None, \
                genotype_format='allele', probs=None, weights=None, \
                verbose=True, profiler=None):
    """
    given a snp file path(s) or a pandas df from read_vcf/read_bgen returns genotypes and probabilities
    snpdata: dataFrame given by read_bgen()
    snps: ?? if None, list(snpdata.index) -> RSIDS
    samples: ?? if None, take the samples from probs -> probs['key]['samples']
            and select only those samples that are unique if len(probs.keys()) > 1 == number of BGEN files read before
    probs: dictionary with probabilities given by read_bgen() 
   Genotype has         
        self.metadata = None
        self.probabilities = dict()
    returns: pd dataframses with everything concatenated?
    """
    if profiler is not None:        
        profiler.enable()

    if genotype_format not in ['allele', '012']:
        raise_error(f'genotype_format should be allele or 012')
    
    if not isinstance(gen, Genotype()):
        raise_error(f'gen is not a Genotype object')

    # TODO: Check weights and get good RSIDs

    # TODO: DONE filter Genotype by user defined rsids (good ones) and samples
    gen = gen.filter(samples=samples, rsids=snps)
    # TODO: DONE consolidate 
    gen = gen.consolidate()  # organise the object os its fastest

    # TODO: DONE Filter out bad SNPS
    # - Bad SPNS are SPNS
    #   - len(REF) != 1 or len(ALT) != 1:
    #   - not (EA has one element and is a string)
     


    snps = gen.rsids
    # if samples and snps:
    #     gen = Filter(gen).by_rsids_and_samples(samples=samples, rsids=snps)
    #     logger.info(f'Filtering by Samples and SNPS')
    # elif samples:
    #     gen = Filter(gen).by_samples(samples=samples)
    #     logger.info(f'no SNPS specified, using all')
    #     logger.info(f'Filtering by Samples')
    #     snps = list(gen.metadata.index)
    # elif snps:
    #     gen = Filter(gen).by_rsids(rsids=snps)
    #     logger.info(f'no samples specified, using all')
    #     logger.info(f'Filtering by SNPS')
    # else:
    #     snps = list(gen.metadata.index)
    #     logger.info(f'no samples specified, using all')
    #     logger.info(f'no SNPS specified, using all')
    
    # TODO assign a value to samples. Uinique samples in all rsids?
    # it works only if all nsps (RSIDs) have the same samples
    samples = gen.probabilities[snps[0]][0]

    # Read weights
    riskscore = None
    if weights is not None:
        weights = read_weights(weights)
        logger.info(f'Calculate riskscore using weights')
        riskscore = pd.DataFrame(0.0, columns=range(1), index=samples, dtype=float)        

    genotype_allele = pd.DataFrame('', columns=snps, index=samples, dtype=str)
    genotype_012 = pd.DataFrame(np.nan, columns=snps, index=samples, dtype=float)
    probability = genotype_012.copy()

    logger.info(f'Calculating genotypes for {len(snps)} SNPs and \
                {len(samples)} samples ... ')

    for snp in alive_progress.alive_it(snps): # iterate RSIDs
        # get SNP info
        parsing_error = False
        REF = gen.metadata['REF'][snp]
        ALT = gen.metadata['ALT'][snp]
        if len(REF) != 1 or len(ALT) != 1:
            parsing_error = True

        if weights is not None:
            EA  = weights['ea'][weights['rsid'] == snp].values
            if len(EA) != 0 and isinstance(EA[0], str) and len(EA[0]) == 1:
                weightSNP = weights['weight'][weights['rsid'] == snp].values[0]
                weightSNP = float(weightSNP)
                EA = EA[0]
            else:
                parsing_error = True
                weightSNP = None

        if parsing_error:
            warn(f'Error parsing EA, REF, or ALT in snp {snp}') 
            # TODO what are now this dataframes?                     
            
            #BUG?
            # genotype_allele.loc[snp, :] = np.nan
            # genotype_012.loc[snp, :] = np.nan
            # probability.loc[snp, :] = pd.Series([[np.nan]*3]*len(samples))
            
            genotype_allele.loc[:, snp] = np.nan
            genotype_012.loc[:, snp] = np.nan
            probability.loc[:, snp] = pd.Series([[np.nan] * 3] * len(samples))
            continue

        # Use probabilites as numpy arrays (from dictionary tuple) instead of pandas dataframes
        GP = gen.probabilities[snp][1]
        imax = np.argmax(GP, axis=1) # get index of largest prob form the three values on each smaple of one RSID
        
        genotype_allele.loc[samples[imax==0], snp] = REF + REF
        genotype_allele.loc[samples[imax==1], snp] = "".join(sorted(REF + ALT))
        genotype_allele.loc[samples[imax==2], snp] = ALT + ALT
        genotype_012.loc[samples, snp] = imax

        probability.loc[:, snp] = pd.Series(GP.tolist()) # TEST

        # todo fix this
        # probability = pd.DataFrame(np.nan, columns=snps, index=samples, dtype=float)
        #probability.loc[snp, GP.index] = GP.values # GP.values, get numpy array 

        if weights is not None and weightSNP is not None:
            # TODO this is quite slow and maybe incorrect
            dosage = GP2dosage(GP, REF, ALT, EA)            
            # whichever samples have a non-nan dosage get added
            # all others become NaN which is good        
            riskscore = riskscore.add(weightSNP * dosage, axis=0)
            # riskscore += weightSNP * dosage

    if profiler is not None:  
        profiler.disable()

    if weights is None:
        return genotype_allele, genotype_012, probability
    else:
        return genotype_allele, genotype_012, riskscore
