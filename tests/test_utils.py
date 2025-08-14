import pandas as pd

from pymgcv.utils import load_rdata_dataframe_from_url


def test_load_rdata_dataframe_from_url():
    url = "https://github.com/mfasiolo/testGam/raw/master/data/Larynx.rda"
    data = load_rdata_dataframe_from_url(url)
    assert isinstance(data, pd.DataFrame)
