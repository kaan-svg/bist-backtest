import math

import numpy as np
import pandas as pd
import pytest

from src.motor import MALIYETSIZ, backtest
from src.olcutler import (dusus_serisi, maksimum_dusus, ozet, sharpe, tablo, yillik_getiri,
                          yillik_volatilite)
from src.strateji import AlTut, HareketliOrtalama


def seri(degerler, baslangic="2024-01-01", siklik="B"):
    return pd.Series(degerler, index=pd.date_range(baslangic, periods=len(degerler), freq=siklik), dtype=float)


def test_yillik_getiri_bir_yilda_ikiye_katlama():
    s = pd.Series([100.0, 200.0], index=pd.to_datetime(["2023-01-01", "2024-01-01"]))
    assert yillik_getiri(s) == pytest.approx(1.0, abs=0.002)   # 365 gun / 365.25


def test_yillik_getiri_iki_yilda_dort_kat():
    s = pd.Series([1.0, 4.0], index=pd.to_datetime(["2022-01-01", "2024-01-01"]))
    assert yillik_getiri(s) == pytest.approx(1.0, abs=0.002)


def test_yillik_getiri_tek_tarih_hata():
    with pytest.raises(ValueError):
        yillik_getiri(seri([1.0]))


def test_volatilite_ve_sharpe_sabit_getiri():
    g = seri([0.001] * 100)
    assert yillik_volatilite(g) == pytest.approx(0)
    assert sharpe(g) is None


def test_sharpe_elle():
    g = seri([0.01, -0.01, 0.02, 0.0])
    beklenen = g.mean() / g.std(ddof=1) * math.sqrt(252)
    assert sharpe(g) == pytest.approx(beklenen)
    assert sharpe(g, risksiz_yillik=0.40) < sharpe(g)   # yuksek TL faizi Sharpe'i dusurur


def test_maksimum_dusus_elle():
    #            0    1    2    3    4    5    6
    s = seri([100, 120, 90, 60, 100, 130, 125])
    md = maksimum_dusus(s)
    assert md["oran"] == pytest.approx(60 / 120 - 1)            # %-50
    assert (md["zirve"], md["dip"], md["toparlanma"]) == (s.index[1], s.index[3], s.index[5])
    assert dusus_serisi(s).iloc[-1] == pytest.approx(125 / 130 - 1)


def test_maksimum_dusus_toparlanmadi():
    md = maksimum_dusus(seri([100, 80, 90]))
    assert md["toparlanma"] is None and md["oran"] == pytest.approx(-0.2)


def test_maksimum_dusus_hic_dusmeyen():
    assert maksimum_dusus(seri([1, 2, 3]))["oran"] == 0


def rastgele_veri(gun=500, tohum=0):
    r = np.random.default_rng(tohum)
    k = 100 * np.exp(np.cumsum(r.normal(0.0005, 0.02, gun)))
    return pd.DataFrame({"acilis": k * np.exp(r.normal(0, 0.01, gun)), "kapanis": k},
                        index=pd.bdate_range("2020-01-01", periods=gun))


def test_ozet_tutarli():
    df = rastgele_veri()
    r = backtest(df, HareketliOrtalama(10, 30))
    o = ozet(r)
    assert o["toplam_getiri"] == pytest.approx(r.toplam_getiri)
    assert o["islem_sayisi"] == r.islem_sayisi
    assert 0 < o["piyasada"] < 1
    assert -1 < o["maksimum_dusus"] <= 0


def test_al_tut_maksimum_dusus_fiyatla_ayni():
    """Maliyetsiz al-tut'un dususu, ikinci gunden itibaren kapanis fiyatinin dususuyle ayni olmali."""
    df = rastgele_veri()
    o = ozet(backtest(df, AlTut(), maliyet=MALIYETSIZ))
    fiyat = df["kapanis"].iloc[1:]
    assert o["maksimum_dusus"] == pytest.approx(maksimum_dusus(fiyat)["oran"], abs=0.02)


def test_tablo():
    df = rastgele_veri()
    metin = tablo([ozet(backtest(df, AlTut())), ozet(backtest(df, HareketliOrtalama(10, 30)))])
    assert "al_tut" in metin and "hareketli_ortalama" in metin and "Sharpe" in metin
