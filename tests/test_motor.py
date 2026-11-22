import numpy as np
import pandas as pd
import pytest

from src.motor import backtest, pozisyon_hesapla
from src.strateji import AlTut, HareketliOrtalama, Strateji


def veri(acilis, kapanis):
    tarih = pd.bdate_range("2024-01-01", periods=len(kapanis))
    return pd.DataFrame({"acilis": acilis, "kapanis": kapanis}, index=tarih, dtype=float)


def rastgele_veri(gun=300, tohum=0):
    r = np.random.default_rng(tohum)
    kapanis = 100 * np.exp(np.cumsum(r.normal(0, 0.02, gun)))
    acilis = kapanis * np.exp(r.normal(0, 0.01, gun))
    return veri(acilis, kapanis)


class Sabit(Strateji):
    """Testler icin: verilen sinyal listesini aynen donduren strateji."""

    def __init__(self, liste):
        super().__init__("sabit", "test", {"liste": tuple(liste)})

    def sinyal(self, df):
        return pd.Series(self.parametreler["liste"], index=df.index, dtype="int8")


def test_sinyal_ertesi_gun_uygulanir():
    s = pd.Series([1, 1, 0, 0, 1], dtype="int8")
    assert list(pozisyon_hesapla(s)) == [0, 1, 1, 0, 0]


def test_elle_hesaplanan_ornek():
    """gun:     0     1     2     3     4
       acilis  100   102   110   108   100
       kapanis 101   105   106   104   99
       sinyal   1     1     0     0     0   -> pozisyon 0 1 1 0 0
       gun 1: alis acilista 102 -> 105        = 105/102
       gun 2: gece 105->110, gun ici 110->106 = 106/105
       gun 3: satis acilista: gece 106 -> 108  = 108/106
    """
    df = veri([100, 102, 110, 108, 100], [101, 105, 106, 104, 99])
    r = backtest(df, Sabit([1, 1, 0, 0, 0]), sermaye=1000)
    beklenen = 1000 * 105 / 102 * 106 / 105 * 108 / 106
    assert r.sermaye.iloc[-1] == pytest.approx(beklenen)
    assert r.toplam_getiri == pytest.approx(108 / 102 - 1)
    islem = r.islemler.iloc[0]
    assert (islem.giris_fiyat, islem.cikis_fiyat, bool(islem.acik)) == (102, 108, False)


def test_kapanistan_islem_yapilmaz():
    """Sinyal gununun kapanisindan alim yapilsaydi gun 1'in tamami (101->105) kazanilirdi.
    Motor acilistan (102) alir; aradaki fark lookahead kazancidir."""
    df = veri([100, 102, 110], [101, 105, 106])
    r = backtest(df, Sabit([1, 0, 0]), sermaye=1)
    assert r.sermaye.iloc[-1] == pytest.approx(105 / 102 * 110 / 105)
    assert r.sermaye.iloc[-1] != pytest.approx(105 / 101 * 110 / 105)


def test_al_tut_ikinci_gun_acilistan_sona_kadar():
    df = rastgele_veri()
    r = backtest(df, AlTut(), sermaye=1)
    assert r.sermaye.iloc[-1] == pytest.approx(df["kapanis"].iloc[-1] / df["acilis"].iloc[1])
    assert r.islem_sayisi == 1 and bool(r.islemler["acik"].iloc[0])


def test_hic_islem_yoksa_sermaye_sabit():
    r = backtest(rastgele_veri(50), Sabit([0] * 50))
    assert (r.sermaye == r.baslangic_sermaye).all() and r.islem_sayisi == 0


def test_islem_getirileri_sermaye_ile_tutarli():
    """Maliyetsizken islem getirilerinin carpimi toplam getiriye esit olmali."""
    df = rastgele_veri()
    r = backtest(df, HareketliOrtalama(5, 20), sermaye=1)
    assert r.islem_sayisi > 3
    assert (1 + r.islemler["getiri"]).prod() == pytest.approx(r.sermaye.iloc[-1])


@pytest.mark.parametrize("strateji", [AlTut(), HareketliOrtalama(5, 20)], ids=str)
def test_motorda_lookahead_yok(strateji):
    """Gelecek degisince gecmis sermaye degismemeli (motor seviyesinde lookahead testi)."""
    df = rastgele_veri()
    tam = backtest(df, strateji).sermaye
    for kesim in range(60, len(df) - 1, 40):
        for faktor in (0.3, 3.0):
            bozuk = df.copy()
            bozuk.iloc[kesim + 1:] *= faktor
            pd.testing.assert_series_equal(tam.iloc[:kesim + 1], backtest(bozuk, strateji).sermaye.iloc[:kesim + 1])


def test_gecersiz_girdiler():
    with pytest.raises(ValueError, match="acilis"):
        backtest(pd.DataFrame({"kapanis": [1.0, 2.0]}), AlTut())
    with pytest.raises(ValueError, match="2 gun"):
        backtest(veri([1.0], [1.0]), AlTut())
    with pytest.raises(ValueError, match="Sermaye"):
        backtest(rastgele_veri(10), AlTut(), sermaye=0)
