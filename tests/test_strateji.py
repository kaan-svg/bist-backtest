import numpy as np
import pandas as pd
import pytest

from src.strateji import STRATEJILER, AlTut, HareketliOrtalama, islem_sayisi, olustur


def fiyat(kapanislar):
    tarih = pd.bdate_range("2024-01-01", periods=len(kapanislar))
    return pd.DataFrame({"kapanis": np.asarray(kapanislar, dtype=float)}, index=tarih)


def rastgele_fiyat(gun=300, tohum=0):
    r = np.random.default_rng(tohum)
    return fiyat(100 * np.exp(np.cumsum(r.normal(0, 0.02, gun))))


# Her stratejinin ornegi; yeni strateji eklendikce buraya eklenir.
ORNEKLER = [AlTut(), HareketliOrtalama(5, 20), HareketliOrtalama(20, 50)]


def lookahead_kontrol(strateji, df):
    """Gelecek degisince gecmis sinyaller degismemeli. Birden fazla kesim noktasinda
    gelecek hem sert dususle (x0.3) hem sert yukselisle (x3) bozulur; tek noktada
    rastgele bozma, gelecegi kullanan bir stratejiyi sans eseri kacirabiliyordu."""
    tam = strateji.sinyal(df)
    for kesim in range(60, len(df) - 1, 20):
        for faktor in (0.3, 3.0):
            bozuk = df.copy()
            bozuk.iloc[kesim + 1:, 0] *= faktor
            pd.testing.assert_series_equal(tam.iloc[:kesim + 1], strateji.sinyal(bozuk).iloc[:kesim + 1],
                                           obj=f"{strateji} kesim={kesim} faktor={faktor}")


@pytest.mark.parametrize("strateji", ORNEKLER, ids=str)
def test_lookahead_yok(strateji):
    lookahead_kontrol(strateji, rastgele_fiyat())


def test_lookahead_testi_hatali_stratejiyi_yakalar():
    """Testin kendisini test et: gelecegi kullanan bir strateji gercekten yakalanmali."""
    class Hileli(HareketliOrtalama):
        def sinyal(self, df):
            return (df["kapanis"].shift(-1) > df["kapanis"]).astype("int8").rename("sinyal")

    with pytest.raises(AssertionError):
        lookahead_kontrol(Hileli(5, 20), rastgele_fiyat())


@pytest.mark.parametrize("strateji", ORNEKLER, ids=str)
def test_sinyal_bicimi(strateji):
    df = rastgele_fiyat()
    s = strateji.sinyal(df)
    assert s.index.equals(df.index)
    assert set(s.unique()) <= {0, 1}


def test_al_tut_hep_hissede():
    s = AlTut().sinyal(rastgele_fiyat(50))
    assert (s == 1).all() and islem_sayisi(s) == 1


def test_hareketli_ortalama_yukselen_ve_dusen_trend():
    yukselen = HareketliOrtalama(2, 4).sinyal(fiyat(range(1, 11)))
    assert list(yukselen) == [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]   # ilk 3 gun uzun ortalama yok
    dusen = HareketliOrtalama(2, 4).sinyal(fiyat(range(10, 0, -1)))
    assert (dusen == 0).all()


def test_hareketli_ortalama_kesisim_gunu():
    # 1..6 yukselis, sonra sert dusus: kisa ortalama uzunun altina iner
    s = HareketliOrtalama(2, 3).sinyal(fiyat([1, 2, 3, 4, 5, 6, 1, 1, 1]))
    assert list(s) == [0, 0, 1, 1, 1, 1, 0, 0, 0]
    assert islem_sayisi(s) == 2


@pytest.mark.parametrize("kisa, uzun", [(50, 20), (20, 20), (0, 20), (5.5, 20), (True, 20)])
def test_hareketli_ortalama_gecersiz_parametre(kisa, uzun):
    with pytest.raises(ValueError):
        HareketliOrtalama(kisa, uzun)


def test_olustur_ve_ad():
    s = olustur("hareketli_ortalama", kisa=10, uzun=30)
    assert str(s) == "hareketli_ortalama(kisa=10, uzun=30)"
    assert set(STRATEJILER) == {"al_tut", "hareketli_ortalama"}
    with pytest.raises(ValueError, match="Bilinmeyen"):
        olustur("sihirli")


def test_bos_veya_kapanissiz_veri():
    with pytest.raises(ValueError):
        AlTut().sinyal(pd.DataFrame({"kapanis": []}))
    with pytest.raises(ValueError, match="kapanis"):
        HareketliOrtalama().sinyal(pd.DataFrame({"fiyat": [1.0]}))
