import io
import os
import glob
import shutil
import subprocess
import requests
import pandas as pd
import numpy as np
from datalad import api as datalad
import alive_progress
from pathlib import Path

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
    return response ## returns a request.Response object, not a JSON object


def datalad_get_chromosome(c,
        source=None,
        imputationdir='imputation',
        path=None):
    """
    get a particular chromosome's (imputed) data
    c: chromosome number, string
    source: datalad source, string (default: None which maps to 'ria+http://ukb.ds.inm7.de#~genetic')
    imputationdir: directory in which the imputation files are stored, string (default: 'imputation')
    path: directory to use for the datalad dataset, string (default: None which maps to '/tmp/genetic')
    returns: list of files, datalad dataset object, list of datalad get output
    """
    if source is None or source == '':
        source="ria+http://ukb.ds.inm7.de#~genetic"

    if path is None or path == '':
        pathOS = os.path.join('/tmp', 'genetic')
        path = Path('/tmp/genetic')

    ds = datalad.clone(source=source, path=path)
    filesOS = glob.glob(os.path.join(ds.path, imputationdir, '*_c' + str(c) + '_*'))
    files = list(Path(ds.path).joinpath(imputationdir).glob('*_c' + str(c) + '_*'))
    getout = ds.get(files)
    return files, ds, getout


def rsid2chromosome(rsids, chromosomes=None):
    """
    get the chromosome of each rsid
    rsids: list of rsids, string or list of strings
    chromosomes: list of chromosomes, string or list of strings
    returns: dataframe with columns 'rsids' and 'chromosomes'
    """
    # if isinstance(rsids, str) and os.path.isfile(rsids):
    if isinstance(rsids, str) and Path(rsids).is_file():
        rsids = pd.read_csv(rsids, header=None, sep='\t')
        if rsids.shape[1] > 1:
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

def rsid2vcf(rsids, outdir,        
        datalad_source="ria+http://ukb.ds.inm7.de#~genetic",
        qctool=None,
        datalad_drop=True,
        datalad_drop_if_got=True,
        datalad_dir=None,
        force=False,
        chromosomes=None,
        chromosomes_use=None):
    """
    get vcf files for a list of rsids
    rsids: list of rsids or a file with rsids, string or list of strings
    datalad_source: datalad source, string (default: 'ria+http://ukb.ds.inm7.de#~genetic')
    qctool: path to qctool, string (default: None, which maps to 'qctool')
    datalad_drop: whether to drop the datalad dataset after getting the files, bool (default: True)
    datalad_drop_if_got: whether to drop files only if downloaded with get, bool (default: True)
    datalad_dir: directory to use for the datalad dataset, string (default: None which maps to '/tmp/genetic')
    force: whether to force re-calculation (based on output file presence), bool (default: False)
    chromosomes: list of chromosomes to process, list of strings (default: None which uses all chromosomes)
    ch_rs: dataframe with columns 'rsids' and 'chromosomes', dataframe (default: None)
    returns: a pandas dataframe with rsid-chromosome pairs and the vcf files are created in the outdir
    """
    # check if qctool is available
    if qctool is None:
        qctool = shutil.which('qctool')

    # if qctool is None or os.path.isfile(qctool) is False:
    if qctool is None or Path(qctool).is_file() is False:
        print('qctool is not available')
        raise

    # if not os.path.exists(outdir):
    if not Path(outdir).exists():
        # os.makedirs(outdir)
        Path(outdir).mkdir()

    print('t6: ',  all([os.listdir(outdir), list(Path(outdir).iterdir())] ))
    # if force is True and os.listdir(outdir):
    if force is True and list(Path(outdir).iterdir()):
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
        # file_vcf = os.path.join(outdir, 'chromosome' + str(ch) + '.vcf')
        file_vcf = Path(outdir, 'chromosome' + str(ch) + '.vcf')
        # if force is False and os.path.isfile(file_vcf):
        if force is False and file_vcf.is_file():
            print('chromosome ' + str(ch) + ' output file exists, skipping: ' + str(file_vcf))
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
        files, ds, getout = datalad_get_chromosome(ch, source=datalad_source, path=datalad_dir)
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
            #name, ext = os.path.splitext(fl)
            name, ext = Path(fl).stem, Path(fl).suffix
            if ext == '.bgen':
                assert file_bgen is None
                file_bgen = fl
            elif ext == '.sample':
                assert file_sample is None
                file_sample = fl

        assert file_bgen is not None and file_sample is not None
        # file_rsids = os.path.join(outdir, 'rsids_chromosome' + str(ch) + '.txt')
        file_rsids = Path(outdir, 'rsids_chromosome' + str(ch) + '.txt')
        df = pd.DataFrame(rs_ch)
        df.to_csv(file_rsids, index=False, header=False)
        
        cmd = qctool + ' -g ' + str(file_bgen) + ' -s ' + str(file_sample) \
              + ' -incl-rsids ' + str(file_rsids)  + ' -og ' + str(file_vcf)
        print('running qctool: ' + cmd  + '\n')
        os.system(cmd)

        if datalad_drop:            
            if datalad_drop_if_got:
                for fi in range(len(getout)):
                    if getout[fi]['status'] == 'ok' and getout[fi]['type'] == 'file':
                        print('datalad: dropping file ' +  str(files[fi]) + '\n')
                        ds.drop(files[fi])
            else:
                print('datalad: dropping all files\n')
                ds.drop(files)

        print('done with chromosome ' + str(ch) + '\n')

    return ch_rs, files, ds

def rsid2vcf_multiple(rsid_files, outdir,
                      qctool=None,
                      datalad_source="ria+http://ukb.ds.inm7.de#~genetic",
                      datalad_dir=None,
                      datalad_drop=True):
    """
    """
    chromosomes = []
    # check if all files are available
    outdirs = [None] * len(rsid_files)
    ch_rs   = [None] * len(rsid_files)
    for i in range(len(rsid_files)):
        print('rsid_file ' + str(i) + ': ' + str(rsid_files[i]))
        # if os.path.isfile(rsid_files[i]) is False: 
        if Path(rsid_files[i]).is_file() is False:
            print('file ' + str(rsid_files[i]) + ' does not exist')
            raise
        # bname = os.path.basename(rsid_files[i])
        # bname = os.path.splitext(bname)[0]
        bname = Path(rsid_files[i]).name
        # outdirs[i] = os.path.join(outdir, bname)
        outdirs[i] = Path(outdir, bname)
        ch_rs[i] = rsid2chromosome(rsid_files[i])
        print(ch_rs[i].head())
        uchromosomes = pd.unique(ch_rs[i]['chromosomes'])
        print(uchromosomes)
        chromosomes = chromosomes + uchromosomes.tolist()
    
    chromosomes = pd.unique(chromosomes)
    print('chromosomes: ' + str(chromosomes))
    print('#rsid_files: ' + str(len(rsid_files)) + '\n')
    for c in range(len(chromosomes)):
        ch = chromosomes[c]
        for i in range(len(rsid_files)):
            print('chromosome ' + str(ch) + ' rsids ' + str(rsid_files[i]) + \
                 ' outdir ' + str(outdirs[i]) + '\n')
            datalad_drop_i = False
            if i == len(rsid_files) - 1:
                datalad_drop_i = datalad_drop
            chrs, files, ds = rsid2vcf(rsids=ch_rs[i]['rsids'].tolist(), outdir=outdirs[i], \
                chromosomes=ch_rs[i]['chromosomes'].tolist(), \
                datalad_source=datalad_source, datalad_dir=datalad_dir, qctool=qctool, \
                chromosomes_use=[ch], force=False, datalad_drop=datalad_drop_i, \
                datalad_drop_if_got=False)

    return outdirs

def read_vcf(path):
    """
    taken shameless from: https://gist.github.com/dceoy/99d976a2c01e7f0ba1c813778f9db744
    Thanks Daichi Narushima
    """
    with open(path, 'r') as f:
        lines = [l for l in f if not l.startswith('##')]
    return pd.read_csv(
        io.StringIO(''.join(lines)),
        dtype={'#CHROM': str, 'POS': int, 'ID': str, 'REF': str, 'ALT': str,
               'QUAL': str, 'FILTER': str, 'INFO': str},
        sep='\t'
    ).rename(columns={'#CHROM': 'CHROM'})


def vcf2genotype(vcf, th=0.9, snps=None, samples=None):
    """
    given a vcf file path or a pandas df from read_vcf returns genotypes
    """
    if isinstance(vcf ,str):
        # assert os.path.isfile(vcf)
        assert Path(vcf).is_file()
        print('reading vcf file: ' + vcf)
        vcf = read_vcf(vcf)
    elif isinstance(vcf, pd.DataFrame):
        pass
    else:
        print("don't know how to handle the input")
        raise

    format = pd.unique(vcf['FORMAT'])
    if len(format) != 1 or format[0] != 'GP':
        print('I can only deal with the GP format')
        raise

    nsnp = vcf.shape[0]
    ncol = vcf.shape[1]
    if samples is None:
        samples = [vcf.columns[i] for i in range(9, ncol)]
    else:
        assert all(sam in list(vcf.columns) for sam in samples)


    if snps is None:
        snps = list(vcf['ID'])
    else:
        assert all(snp in list(vcf['ID']) for snp in snps)

    labels = pd.DataFrame(index=range(len(snps)), columns=range(len(samples)))
    labels.index = snps
    labels.columns = samples
    snps_index = [snps.index(snp) for snp in snps]
    print('calculating genotypes for ' + str(len(snps_index)) + ' SNPs and ' + \
          str(len(samples)) + ' samples ... ')
    for snp in snps_index:
        REF = vcf['REF'][snp]
        ALT = vcf['ALT'][snp]
        for sam in alive_progress.alive_it(samples):
            try:
                GP = vcf[sam][snp]
                GP = [float(x) for x in GP.split(',')]
            except:
                print('error reading sample ' + str(sam) + \
                      ' snp ' + str(snp) + ': ' + str(GP))
                #raise                
                continue
            assert len(GP) == 3
            # quirky way to get argmax, can simply use np.argmax
            f = lambda i: GP[i]
            GT = max(range(len(GP)), key=f)
            if GP[GT] >= th:
                if GT == 0:
                    labels[sam][snp] = REF + REF
                elif GT == 1:
                    labels[sam][snp] = REF + ALT
                else:
                    labels[sam][snp] = ALT + ALT

    return labels


def vcf2prs(vcf_files, weight_file, samples=None, outfile=None):
    """
    given a list of vcf files and a file with weights, returns a pandas df with
    the polygenic risk scores for each sample
    vcf_files: list of vcf files or a directory with vcf files, str or list of str
    weight_file: file with weights (must contain header and columns snpid/rsid, ea and weight), str
    samples: list of samples to use, list of str (default: None, which means all samples)    
    outfile: file to write the output, str (default: None, which means no file written)
    returns: polygenic risk score for each sample, pandas df
    """
    # assert os.path.isfile(weight_file)
    assert Path(weight_file).is_file()

    # if not isinstance(vcf_files, list) and os.path.isdir(vcf_files):
    if not isinstance(vcf_files, list) and Path(vcf_files).is_dir():
        vcf_filesG = glob.glob(vcf_files + '/*.vcf')
        vcf_files = list(Path(vcf_files).glob('*.vcf'))
        print('t6: ', vcf_filesG == vcf_files)        

    assert isinstance(vcf_files, list)
    assert len(vcf_files) > 0

    weights = pd.read_csv(weight_file, sep='\t')
    weights.columns = [x.lower() for x in weights.columns]

    assert 'ea' in weights.columns
    assert 'weight' in weights.columns
    if 'snpid' in weights.columns:
        rsidcol = 'snpid'
    elif 'rsid' in weights.columns:
        rsidcol = 'rsid'
    else:
        print('no rsid column in weight file')
        raise
    rsids_weights = weights[rsidcol].tolist()
    assert len(rsids_weights) == len(set(rsids_weights))     
    print('weight file contains ' + str(len(rsids_weights)) + ' rsids')
    
    # read all the vcf_files
    print('reading ' + str(len(vcf_files)) + ' vcf files... ')
    vcf = pd.DataFrame()
    for vf in alive_progress.alive_it(vcf_files):
        vcf = vcf.append(read_vcf(vf))
    # qctool vcf file ID conrains rsID,loc_info
    # convert to rsID and set it as index to later use
    rsids_vcf = [x.split(',')[0] for x in vcf['ID']]
    vcf.index = rsids_vcf

    # make sure all rsIDs are available
    assert set(rsids_weights).issubset(set(rsids_vcf))

    nsnp = vcf.shape[0]
    ncol = vcf.shape[1]
    if samples is None:
        samples = [vcf.columns[i] for i in range(9, ncol)]
    else:
        assert all(sam in list(vcf.columns) for sam in samples)
    
    print('calculating PRS for ' + str(len(samples)) + ' samples ... ')
    PRS = pd.DataFrame(0, index=range(1), columns=range(len(samples)))
    PRS.columns = samples
    for snp in rsids_weights:
        REF = vcf['REF'][snp]
        ALT = vcf['ALT'][snp]
        EA  = weights['ea'][weights[rsidcol] == snp].values[0]
        assert isinstance(EA, str)
        weightSNP = weights['weight'][weights[rsidcol] == snp].values[0]
        assert isinstance(weightSNP, float)
        for sam in alive_progress.alive_it(samples):
            try:
                GP = vcf[sam][snp]
                GP = [float(x) for x in GP.split(',')]
            except:
                print('error reading sample ' + str(sam) + \
                      ' snp ' + str(snp) + ': ' + str(GP))
                #raise
                PRS[sam] = np.nan
                continue
            assert len(GP) == 3
            pHomoREF = GP[0]
            pHeteroz = GP[1]
            pHomoALT = GP[2]
            if EA == REF:
                dosage = pHeteroz + 2*pHomoREF
            elif EA == ALT:
                dosage = pHeteroz + 2*pHomoALT
            else:
                print('SNP ' + snp + ' ALT ' + ALT + ' or REF ' + REF + \
                      ' do not match EA ' + EA)
                raise        
            PRS[sam] += dosage*weightSNP

    if outfile is not None:
        if isinstance(outfile, str):        
            print('writing file: ' + outfile)
            try:
                PRS.to_csv(outfile, sep='\t')
            except:
                print('error writing file: ' + outfile)                
        else:
            print('outfile argument is not a string, no file written')

    return PRS
