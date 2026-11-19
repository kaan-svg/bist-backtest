"""Veri katmani testleri. Internet gerekmez: sahte indirici kullanilir."""
import numpy as np
import pandas as pd
import pytest

from src.veri import SUTUNLAR, _bicimle, temizle, yukle


def yahoo_cerceve(gun=10, baslangic="2024-01-01", cok_seviyeli=False):
    tarih = pd.bdate_range(baslangic, periods=gun)
    kapanis = np.linspace(100, 109, gun)
    df = pd.DataFrame({"Open": kapanis - 1, "High": kapanis + 1, "Low": kapanis - 2,
                       "Close": kapanis, "Volume": 1000}, index=tarih)
    if cok_seviyeli:
        df.columns = pd.MultiIndex.from_product([df.columns, ["THYAO.IS"]])
    return df


def standart(df):
    return _bicimle(df)


def test_bicimle_sutun_adlari():
    df = standart(yahoo_cerceve())
    assert list(df.columns) == SUTUNLAR and df.index.name == "tarih"


def test_bicimle_cok_seviyeli_sutunlar():
    assert list(standart(yahoo_cerceve(cok_seviyeli=True)).columns) == SUTUNLAR


def test_bicimle_eksik_sutun():
    with pytest.raises(ValueError, match="Volume"):
        _bicimle(yahoo_cerceve().drop(columns="Volume"))


def test_temizle_siralar_ve_tekrari_siler():
    df = standart(yahoo_cerceve())
    karisik = pd.concat([df.iloc[5:], df.iloc[:5], df.iloc[[3]]])
    temiz, uyarilar = temizle(karisik)
    assert temiz.index.is_monotonic_increasing and len(temiz) == 10
    assert any("tekrar" in u for u in uyarilar)


def test_temizle_gecersiz_kapanis():
    df = standart(yahoo_cerceve())
    df.iloc[2, df.columns.get_loc("kapanis")] = np.nan
    df.iloc[4, df.columns.get_loc("kapanis")] = 0
    temiz, uyarilar = temizle(df)
    assert len(temiz) == 8 and any("gecersiz" in u for u in uyarilar)


def test_temizle_yuksek_dusukten_kucuk():
    df = standart(yahoo_cerceve())
    df.iloc[1, df.columns.get_loc("yuksek")] = 1
    assert len(temizle(df)[0]) == 9


def test_temizle_bosluk_uyarisi():
    df = standart(yahoo_cerceve(gun=20))
    bosluklu = pd.concat([df.iloc[:5], df.iloc[15:]])
    _, uyarilar = temizle(bosluklu)
    assert any("boslugu" in u for u in uyarilar)


def test_temizle_bos_veri_hata():
    df = standart(yahoo_cerceve(gun=2))
    df["kapanis"] = np.nan
    with pytest.raises(ValueError, match="veri kalmadi"):
        temizle(df)


def test_yukle_onbellek(tmp_path):
    cagrilar = []

    def sahte(sembol, bas, bit):
        cagrilar.append(sembol)
        return yahoo_cerceve()

    df1, _ = yukle("THYAO.IS", "2024-01-01", "2024-02-01", indirici=sahte, klasor=tmp_path)
    df2, _ = yukle("THYAO.IS", "2024-01-01", "2024-02-01", indirici=sahte, klasor=tmp_path)
    assert cagrilar == ["THYAO.IS"]                     # ikinci seferde onbellekten
    pd.testing.assert_frame_equal(df1, df2, check_freq=False)
    yukle("THYAO.IS", "2024-01-01", "2024-02-01", indirici=sahte, klasor=tmp_path, yenile=True)
    assert len(cagrilar) == 2


def test_yukle_bos_veri_anlasilir_hata(tmp_path):
    with pytest.raises(ValueError, match=r"\.IS"):
        yukle("THYAO", "2024-01-01", "2024-02-01", indirici=lambda *a: pd.DataFrame(), klasor=tmp_path)
