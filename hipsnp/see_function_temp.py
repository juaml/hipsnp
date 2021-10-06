import pytest
import os
from requests.models import Response
import hipsnp as hps
import json
import pandas as pd
import datalad.api as dl
import tempfile
import numpy as np

def try_rsid2snp():
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen'
    rsids = ['RSID_101']
    chromosomes = ['1']
    qctool = '/home/oportoles/Apps/qctool_v2.0.6-Ubuntu16.04-x86_64/qctool'

    with tempfile.TemporaryDirectory() as tempdir:

        ch_rs, files, dataL = hps.rsid2snp(rsids,
                                           outdir=tempdir,
                                           datalad_source=source,
                                           qctool=qctool,
                                           datalad_drop=True,
                                           datalad_drop_if_got=True,
                                           data_dir=tempdir,
                                           force=False,
                                           chromosomes=chromosomes,
                                           chromosomes_use=None,
                                           outformat='bgen')
    return ch_rs, files, dataL
   

if __name__ == "__main__":
    # test_datalad_get_chromosome_return_DataladType()
    ch_rs, files, dataL = try_rsid2snp()
    np.savez('/home/oportoles/Documents/MyCode/hipsnp/test_data/output_rsi2snp.npz', ch_rs=ch_rs, files=files, dataL=dataL, allow_pickle=True)