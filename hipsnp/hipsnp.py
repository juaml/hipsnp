import io
import os
import glob
import shutil
import subprocess
import requests
import pandas as pd
import numpy as np
from datalad import api as datalad
from alive_progress import alive_it
from bgen_reader import open_bgen


def get_chromosome(c,
        datalad_source=None,
        imputationdir='imputation',
        data_dir=None):
    """
    get a particular chromosome's (imputed) data
    c: chromosome number, string
    source: datalad source, string (default: None which maps to 'ria+http://ukb.ds.inm7.de#~genetic')
    imputationdir: directory in which the imputation files are stored, string (default: 'imputation')
    data_dir: directory to use for the datalad dataset, string (default: None which maps to '/tmp/genetic')
    returns: list of files, datalad dataset object, list of datalad get output
    """
    if datalad_source is None or datalad_source == '':
        files = glob.glob(os.path.join(ds.path, imputationdir, '*_c' + str(c) + '_*'))
        ds = None
        getout = ['datalad not used']*len(files)
    else:
        if data_dir is None or data_dir == '':
            data_dir = os.path.join('/tmp', 'genetic')

        ds = datalad.clone(source=datalad_source, path=data_dir)
        files = glob.glob(os.path.join(ds.path, imputationdir, '*_c' + str(c) + '_*'))
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
        raise
    
    url = 'http://rest.ensembl.org/variation/human/' + rsid + '?content-type=application/json'
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
            if isinstance(rsids.iloc[0,1], str):
                rsids.drop(index=0, inplace=True)
            chromosomes = list(rsids.iloc[:,1])
            chromosomes = [str(c) for c in chromosomes]
        rsids = list(rsids.iloc[:,0])
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


def rsid2snp(rsids, outdir,
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
    datalad_source: datalad source, string (default: 'ria+http://ukb.ds.inm7.de#~genetic')
    qctool: path to qctool, string (default: None, which maps to 'qctool')
    datalad_drop: whether to drop the datalad dataset after getting the files, bool (default: True)
    datalad_drop_if_got: whether to drop files only if downloaded with get, bool (default: True)
    data_dir: directory to use for the (datalad) dataset, string (default: None which maps to '/tmp/genetic')
    force: whether to force re-calculation (based on output file presence), bool (default: False)
    chromosomes: list of chromosomes to process, list of strings (default: None which uses all chromosomes)
    ch_rs: dataframe with columns 'rsids' and 'chromosomes', dataframe (default: None)
    returns: a pandas dataframe with rsid-chromosome pairs and the vcf files are created in the outdir
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
            print ('skipping chromosome ' + str(ch), ' not in the use list')
            continue
        file_out = os.path.join(outdir, 'chromosome_' + str(ch) + '.' + outformat)
        if force is False and os.path.isfile(file_out):
            print('chromosome ' + str(ch) + ' output file exists, skipping: ' + str(file_out))
            continue

        ind = [i for i, x in enumerate(ch_rs['chromosomes']) if x == uchromosomes[c]]
        rs_ch = [rsids[i] for i in ind]
        print('chromosome ' + str(ch) + ' with ' + str(len(rs_ch)) + ' rsids\n')
        if len(rs_ch) == 0:
            continue
        
        if len(rs_ch) < 11:
            print('rsids: ' + str(rs_ch) + '\n')

        # get the data
        print('datalad: getting files')
        files, ds, getout = get_chromosome(ch, datalad_source=datalad_source, data_dir=data_dir)
        for fi in range(len(getout)):
            status = getout[fi]['status']
            print('datalad: status ' + str(status) + ' file ' + str(files[fi]))
            if status != 'ok' and status != 'notneeded':
                print('datalad: error getting file ' \
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
        file_rsids = os.path.join(outdir, 'rsids_chromosome' + str(ch) + '.txt')
        df = pd.DataFrame(rs_ch)
        df.to_csv(file_rsids, index=False, header=False)
        
        cmd = qctool + ' -g ' + file_bgen + ' -s ' + file_sample \
              + ' -incl-rsids ' + file_rsids  + ' -og ' + file_out
        if outformat == 'bgen':
            cmd += ' -ofiletype bgen_v1.2 -bgen-bits 8'

        print('running qctool: ' + cmd  + '\n')
        os.system(cmd)

        if datalad_drop:
            # must use relative paths???
            common_prefix = os.path.commonprefix([files[0], ds.path])
            files_rel = [os.path.relpath(path, common_prefix) for path in files]
            if datalad_drop_if_got:
                for fi in range(len(getout)):
                    if getout[fi]['status'] == 'ok' and getout[fi]['type'] == 'file':
                        print('datalad: dropping file ' +  str(files_rel[fi]) + '\n')
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
            print('chromosome ' + str(ch) + ' rsids ' + str(files[i]) + \
                 ' outdir ' + str(outdirs[i]) + '\n')
            datalad_drop_i = False
            if i == len(files) - 1:
                datalad_drop_i = datalad_drop
            chrs, chfiles, ds = rsid2snp(rsids=ch_rs[i]['rsids'].tolist(), outdir=outdirs[i], \
                chromosomes=ch_rs[i]['chromosomes'].tolist(), \
                datalad_source=datalad_source, data_dir=data_dir, qctool=qctool, \
                chromosomes_use=[ch], force=False, datalad_drop=datalad_drop_i, \
                datalad_drop_if_got=False, outformat=outformat)

    return outdirs


def read_bgen(files, rsids_as_index=True, no_neg_samples=False, \
            join='inner', verify_integrity=False, \
            probs_in_pd=False, verbose=True):
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
    for f in alive_it(files):
        if verbose:
            print('reading ' + f)
        bgen = open_bgen(f, verbose=False)
        # sanity checks
        # we can only deal with biallelic variants
        nalleles = np.unique(bgen.nalleles)
        assert len(nalleles) == 1 and nalleles[0] == 2
        probs = bgen.read() # read the probabilities

        tmp = pd.DataFrame(index=range(len(bgen.rsids)))
        if rsids_as_index:
            tmp.index = bgen.rsids

        if probs_in_pd:
            # this is extremely slow
            print('putting probabilities in the dataframe')
            for j in range(probs.shape[1]): #snps
                for i in range(probs.shape[0]): # samples           
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
        tmp = tmp.assign(CHROM=bgen.chromosomes, POS=bgen.positions, ID=bgen.ids)
        tmp['FORMAT'] = 'GP'

        myjoin = join
        if f == files[0]:
            myjoin = 'outer'

        snpdata = pd.concat([snpdata, tmp], join=myjoin, axis=0, verify_integrity=verify_integrity)
        tmp = None

    return snpdata, probsdata


def read_weights(weights):
    """
    read weights from a file
    """
    if isinstance(weights, str):
        assert os.path.isfile(weights)
        weights = pd.read_csv(weights, sep='\t', comment='#')
    weights.columns = [x.lower() for x in weights.columns]    
    weights.rename(columns={'snpid':'rsid', 'chr_name':'chr', \
        'effect_allele':'ea', 'effect_weight':'weight'}, inplace = True)    

    assert 'ea' in weights.columns
    assert 'weight' in weights.columns
    assert 'rsid' in weights.columns    
    rsids = weights['rsid'].tolist()
    # make sure that all rsids are unique
    assert len(rsids) == len(set(rsids))
    return weights


def read_vcf(files, rsids_as_index=True, format=['GP', 'GT:GP'], \
            no_neg_samples=False, \
            join='inner', verify_integrity=False, verbose=True):
    """
    modified shamelessly from: https://gist.github.com/dceoy/99d976a2c01e7f0ba1c813778f9db744
    Thanks Daichi Narushima
    """
    # todo: keep lines starting with ## as metadata in attrs
    # pandas does not support more than 1 char as comment, e.g. '##'
    # some VCF files might not have FORMAT column
    if isinstance(files, str):
        files = [files]

    # make sure that files exist
    for vf in files:
        assert os.path.isfile(vf)

    # read all the files
    if verbose:
        print('reading ' + str(len(files)) + ' vcf files... ')
    snpdata = pd.DataFrame()
    for vf in alive_it(files):
        if verbose:
            print('reading ' + vf)

        with open(vf, 'r') as f:
            lines = [l for l in f if not l.startswith('##')]
        
        tmp = pd.read_csv(io.StringIO(''.join(lines)), sep='\t', \
                dtype=str)
              #dtype={'#CHROM': str, 'POS': int, 'ID': str, 'REF': str, 'ALT': str,
              #'QUAL': str, 'FILTER': str, 'INFO': str})
        tmp.rename(columns={'#CHROM': 'CHROM'}, inplace=True)

        # check format
        if format is not None:
            fmt = pd.unique(tmp['FORMAT'])
            if len(fmt) != 1:
                print('multiple formats in file: ' + str(vf))
                raise
            fmt = fmt[0]
            if fmt not in format:
                print('I can deal with formats: ' + str(format), ' but got ' + str(fmt))
                print('file: ' + str(vf))
                raise

        # qctool vcf file ID conrains rsID,loc_info
        # convert to rsID and set it as index to later use
        if rsids_as_index:
            rsids = [x.split(',')[0] for x in tmp['ID']]
            tmp.index = rsids
        
        if no_neg_samples:
            ncol = tmp.shape[1]
            samples = tmp.columns.to_list()[9:ncol]
            samples = np.array([int(s) for s in samples])
            if np.any(samples < 0):
                print('negative samples found: ' + vf)
                raise
        
        myjoin = join
        if vf == files[0]:
            myjoin = 'outer'

        snpdata = pd.concat([snpdata, tmp], join=myjoin, axis=0, verify_integrity=verify_integrity)
        tmp = None

    return snpdata, None # None for compatibility with read_bgen


def snp2genotype(snpdata, th=0.9, snps=None, samples=None, \
                genotype_format='allele', probs=None, weights=None, \
                verbose=True, profiler=None):
    """
    given a snp file path(s) or a pandas df from read_vcf/read_bgen returns genotypes and probabilities
    """
    if profiler is not None:        
        profiler.enable()

    assert isinstance(genotype_format, str)
    assert genotype_format in ['allele', '012']

    if not isinstance(snpdata, pd.DataFrame):
        print("don't know how to handle the input")
        print('please use read_vcf or read_bgen to get the required input')
        raise

    nsnp = snpdata.shape[0]
    ncol = snpdata.shape[1]
    if samples is None:
        if verbose:
            print('no samples specified, using all')
        if probs is None:
            samples = [snpdata.columns[i] for i in range(9, ncol)]
        else:
            pkeys = list(probs.keys())
            samples = probs[pkeys[0]]['samples']
            for pk in range(1, len(pkeys)):
                samples = np.append(samples, probs[pkeys[pk]]['samples'])
                samples = np.unique(samples)
    else:
        if probs is None:
            assert all(sam in snpdata.columns for sam in samples)

    if snps is None:
        snps = list(snpdata.index)
    #else:
    #    assert all(snp in list(snp.index) for s in snps)

    riskscore = None
    if weights is not None:
        weights = read_weights(weights)
        if verbose:
            print('will calculate riskscore using weights')
            print(weights.head())
        riskscore = pd.DataFrame(0.0, columns=range(1), index=samples, dtype=float)        

    genotype_allele = pd.DataFrame('', columns=snps, index=samples, dtype=str)
    genotype_012 = pd.DataFrame(np.nan, columns=snps, index=samples, dtype=float)
    probability = genotype_012.copy()
    print('calculating genotypes for ' + str(len(snps)) + ' SNPs and ' + \
          str(len(samples)) + ' samples ... ')
    for snp in alive_it(snps):
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
            probability.loc[snp, :] = pd.Series([[np.nan]*3]*len(samples))
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
                        probs_pk = probs[pk]['probs'][:, snpidx, :]
                        probs_pk = probs_pk[index_mask]               
                        # probs is samples x snps x 3
                        GP = pd.DataFrame(probs_pk, index=index[index_mask],
                                          columns=range(3))
                        # keep what we need
                        # todo check if this reorders
                        #GP = GP.reindex(samples).dropna()
                        #break
            else:
                # todo: vectorize this
                GP = pd.DataFrame(index=samples, columns=range(3))
                for sam in samples:
                    try:
                        gt, gp = parse_GTGP(snpdata[sam][snp])
                        GP.loc[sam,:] = gp
                    except Exception as e:
                        if verbose:
                            print('error parsing snp ' + str(snp))
                            print(e)
                        genotype[sam][snp] = np.nan
                        probability[sam][snp] = [np.nan]*3
                        continue
        except Exception as e:
            print(e)
        
        # use GP to calculate genotype        
        imax = np.argmax(GP.values, axis=1)
        genotype_allele.loc[GP.index[imax == 0], snp] = REF + REF
        genotype_allele.loc[GP.index[imax == 1], snp] = "".join(sorted(REF + ALT))
        genotype_allele.loc[GP.index[imax == 2], snp] = ALT + ALT
        genotype_012.loc[GP.index, snp] = imax

        # todo fix this
        #probability.loc[snp, GP.index] = GP.values

        if weights is not None and weightSNP is not None:
            # todo this is quite slow and maybe incorrect
            #print('SNP ' + snp + ' EA ' + EA + ' REF ' + REF +\
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
        dosage = GP.iloc[:,1] + 2*GP.iloc[:,0]
    elif EA == ALT:
        dosage = GP.iloc[:,1] + 2*GP.iloc[:,2]
    else:
        print('SNP ' + snp + ' ALT ' + ALT + ' or REF ' + REF + \
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
        except:
            GP = [np.nan, np.nan, np.nan]
    GP = np.array(GP)
    return GT, GP
