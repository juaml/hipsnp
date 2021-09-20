from hipsnp import hipsnp
import pytest
import hipsnp.hipsnp as hps
import json
import pandas as pd


mock_rsid = 'rs699'

def validateJSON(jsonData):
    """attemts to open a JSON var"""
    try:
        json_object = json.loads(jsonData)
    except ValueError as e:
        return False
    return True


def test_JSON_format_ensembl_human_rsid():
    """test output is in JSON format"""
    mock_rsid = 'rs699'
    outJSON = hps.ensembl_human_rsid(mock_rsid)
    assert validateJSON(outJSON)


def validatePANDAStype(pdData):
    """Data is a pandas DataFrame with fields 'chromosomes' and 'rsids' of type str"""

    assert isinstance(pdData, pd.core.frame.DataFrame)

def test_rsid2chromosome_has_pandas_format():
    mock_rsid = 'rs699'
    outPANDAS = hps.rsid2chromosome(mock_rsid)
    validatePANDAStype(outPANDAS)

def valiadtePANDAS_has_RSDIandCROMOSOM(pdData, refColFields):
    outFields = [field for field in pdData.columns]
    assert refColFields.sort() == outFields.sort()

def test_rsid2chromosome_has_RSIDandCROMOSOM():
    mock_rsid = 'rs699'
    refColFields = ['rsids','chromosomes']
    outPANDAS = hps.rsid2chromosome(mock_rsid)

    valiadtePANDAS_has_RSDIandCROMOSOM(outPANDAS, refColFields)


def test_rsid2chromosome_has_list_of_RSIDandCROMOSOM():
    mock_rsid = ['rs699', 'rs698']
    refColFields = ['rsids','chromosomes']
    outPANDAS = hps.rsid2chromosome(mock_rsid)

    valiadtePANDAS_has_RSDIandCROMOSOM(outPANDAS, refColFields)




