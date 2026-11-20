"""Stratejiler: fiyat verisinden gunluk hedef pozisyon uretir.

Her strateji ayni arayuzu kullanir:
    sinyal(df) -> pd.Series   (df ile ayni tarihler, degerler 0 = nakit, 1 = hissede)

KURAL (lookahead yasagi): t gunundeki sinyal sadece t gunu ve oncesinin verisiyle
hesaplanir. Sinyal t gunu kapanistan sonra bilinir, bu yuzden islem en erken t+1
gunu yapilabilir; bu kaydirmayi backtest motoru yapacak (Gun 3). Stratejiler
kaydirma yapmaz.

Bu kurali tests/test_strateji.py her strateji icin otomatik test eder: gelecekteki
fiyatlar degistirildiginde gecmis sinyaller degismemeli.
"""
from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class Strateji:
    ad: str
    aciklama: str
    parametreler: dict = field(default_factory=dict)

    def sinyal(self, df):
        raise NotImplementedError

    def __str__(self):
        p = ", ".join(f"{k}={v}" for k, v in self.parametreler.items())
        return f"{self.ad}({p})" if p else self.ad


def _kontrol(df):
    if "kapanis" not in df.columns:
        raise ValueError("Veride 'kapanis' sutunu yok")
    if df.empty:
        raise ValueError("Bos veri")


def _pozitif_tam_sayi(deger, ad):
    if not isinstance(deger, int) or isinstance(deger, bool) or deger < 1:
        raise ValueError(f"{ad} pozitif tam sayi olmali, verilen: {deger!r}")


class AlTut(Strateji):
    """Ilk gun al, hic satma. Diger stratejilerin gecmesi gereken taban cizgi."""

    def __init__(self):
        super().__init__("al_tut", "Ilk gun al, sonuna kadar tut")

    def sinyal(self, df):
        _kontrol(df)
        return pd.Series(1, index=df.index, name="sinyal", dtype="int8")


class HareketliOrtalama(Strateji):
    """Kisa hareketli ortalama uzun olanin ustundeyse hissede, degilse nakitte.

    Ilk `uzun` gun ortalama hesaplanamadigi icin nakitte kalinir.
    """

    def __init__(self, kisa=20, uzun=50):
        _pozitif_tam_sayi(kisa, "kisa")
        _pozitif_tam_sayi(uzun, "uzun")
        if kisa >= uzun:
            raise ValueError(f"kisa ({kisa}) uzun'dan ({uzun}) kucuk olmali")
        super().__init__("hareketli_ortalama", "Kisa ortalama > uzun ortalama ise hissede",
                         {"kisa": kisa, "uzun": uzun})

    def sinyal(self, df):
        _kontrol(df)
        kisa = df["kapanis"].rolling(self.parametreler["kisa"]).mean()
        uzun = df["kapanis"].rolling(self.parametreler["uzun"]).mean()
        return (kisa > uzun).astype("int8").rename("sinyal")   # NaN > x -> False -> 0


STRATEJILER = {"al_tut": AlTut, "hareketli_ortalama": HareketliOrtalama}


def olustur(ad, **parametreler):
    if ad not in STRATEJILER:
        raise ValueError(f"Bilinmeyen strateji: {ad}. Mevcut: {', '.join(STRATEJILER)}")
    return STRATEJILER[ad](**parametreler)


def islem_sayisi(sinyal):
    """Pozisyon degisim sayisi (alis + satis). Ilk gun 1 ise o da bir alistir."""
    return int(sinyal.diff().fillna(sinyal.iloc[0]).abs().sum())


if __name__ == "__main__":
    import sys

    from src.veri import yukle

    sembol = sys.argv[1] if len(sys.argv) > 1 else "THYAO.IS"
    df, _ = yukle(sembol, "2018-01-01", "2025-12-31")
    for s in (AlTut(), HareketliOrtalama(20, 50), HareketliOrtalama(50, 200)):
        sin = s.sinyal(df)
        print(f"{str(s):<36} hissede gecen gun: %{100 * sin.mean():.0f}   islem: {islem_sayisi(sin)}")
