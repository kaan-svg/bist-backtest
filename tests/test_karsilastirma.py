import numpy as np
import pandas as pd
import pytest

from src.karsilastirma import beta_korelasyon, faiz_egrisi, hizala, karsilastir, olcek_duzelt
from src.olcutler import yillik_getiri
from src.strateji import AlTut, HareketliOrtalama


def fiyat(kapanis, baslangic="2020-01-01"):
    k = np.asarray(kapanis, dtype=float)
    return pd.DataFrame({"acilis": k, "yuksek": k, "dusuk": k, "kapanis": k, "hacim": 1.0},
                        index=pd.bdate_range(baslangic, periods=len(k)))


def rastgele(gun=400, tohum=0, kayma=0.0005):
    r = np.random.default_rng(tohum)
    k = 100 * np.exp(np.cumsum(r.normal(kayma, 0.015, gun)))
    df = fiyat(k)
    df["acilis"] = k * np.exp(r.normal(0, 0.005, gun))
    return df


def test_olcek_duzelt_iki_sifir_atma():
    """BIST 100'de 2020'de oldugu gibi: endeks bir gunde 100'e bolunuyor."""
    df = fiyat([100_000, 101_000, 1_020, 1_030])
    duz, uyarilar = olcek_duzelt(df)
    assert list(duz["kapanis"]) == pytest.approx([1_000, 1_010, 1_020, 1_030])
    assert duz["hacim"].tolist() == [1.0] * 4 and len(uyarilar) == 1


def test_olcek_duzelt_normal_veriye_dokunmaz():
    df = rastgele()
    duz, uyarilar = olcek_duzelt(df)
    pd.testing.assert_frame_equal(df, duz)
    assert not uyarilar


def test_hizala_ortak_gunler():
    a = fiyat(range(1, 11))
    b = fiyat(range(1, 11), baslangic="2020-01-06")
    x, y = hizala(a, b)
    assert x.index.equals(y.index) and len(x) == 7   # 6-14 Ocak 2020 arasi is gunleri


def test_hizala_ortak_gun_yok():
    with pytest.raises(ValueError, match="Ortak"):
        hizala(fiyat([1, 2]), fiyat([1, 2], baslangic="2030-01-01"))


def test_faiz_egrisi_bir_yil():
    tarih = pd.DatetimeIndex(["2023-01-01", "2023-07-02", "2024-01-01"])
    assert faiz_egrisi(tarih, 0.40).iloc[-1] == pytest.approx(1.40, abs=0.001)


def test_faiz_egrisi_yillik_getiri_ile_tutarli():
    """Yilda kac islem gunu olursa olsun, olculen yillik getiri girilen faize esit olmali."""
    for gun in (300, 700):
        egri = faiz_egrisi(pd.bdate_range("2020-01-01", periods=gun), 0.30)
        assert yillik_getiri(egri) == pytest.approx(0.30, abs=1e-9)


def test_beta_kendisi_ve_iki_kati():
    e = rastgele()["kapanis"].pct_change().dropna()
    assert beta_korelasyon(e, e) == pytest.approx((1.0, 1.0))
    assert beta_korelasyon(2 * e, e)[0] == pytest.approx(2.0)
    assert beta_korelasyon(e * 0, e) == (pytest.approx(0.0), None)   # hic islem yok


def test_karsilastir_ayni_hisse_ve_endeks():
    """Hisse ile endeks ayniysa al-tut'un endekse gore farki 0, betasi 1 olmali."""
    df = rastgele()
    ozetler, _ = karsilastir(df, df.copy(), [AlTut()])
    al_tut, endeks = ozetler
    assert al_tut["endekse_gore"] == pytest.approx(0, abs=1e-12)
    assert al_tut["beta"] == pytest.approx(1.0)
    assert endeks["strateji"] == "BIST 100 al-tut"


def test_karsilastir_faiz_farki():
    df = rastgele()
    ozetler, faiz_yillik = karsilastir(df, rastgele(tohum=1), [AlTut(), HareketliOrtalama(10, 30)], faiz=0.30)
    assert faiz_yillik == pytest.approx(0.30, abs=0.01)
    for o in ozetler:
        assert o["faize_gore"] == pytest.approx(o["yillik_getiri"] - faiz_yillik)


def test_nakitte_kalan_stratejinin_betasi_dusuk():
    df, endeks = rastgele(tohum=2), rastgele(tohum=3)
    ozetler, _ = karsilastir(df, endeks, [AlTut(), HareketliOrtalama(20, 60)])
    assert abs(ozetler[1]["beta"]) <= abs(ozetler[0]["beta"]) + 0.05
