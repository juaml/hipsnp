import hipsnp.hipsnp as hps
import pandas as pd


def test_rsid2chromosome_has_list_of_RSIDandCROMOSOM():
    mock_rsid = ['rs699', 'rs698']
    refColFields = ['rsids','chromosomes']

    outPANDAS = hps.rsid2chromosome(mock_rsid)
    return outPANDAS

test_rsid2chromosome_has_list_of_RSIDandCROMOSOM()