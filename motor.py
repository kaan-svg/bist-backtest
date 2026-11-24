"""Backtest motoru: stratejinin sinyallerini gunluk getiriye ve sermaye egrisine cevirir.

Zamanlama (en kolay hata yapilan yer):
  - Sinyal t gunu KAPANISTA hesaplanir (o gunun kapanis fiyati ancak o zaman bilinir).
  - Islem t+1 gunu ACILISTA yapilir. t gununun kapanisindan islem yapmak, o fiyati
    onceden bilmek demektir: bu bir lookahead hatasidir ve sonuclari iyi gosterir.

Gunluk getiri iki parcaya bolunur:
  gece    = acilis(t) / kapanis(t-1) - 1   -> dun aksam hissedeysek kazanilir
  gun ici = kapanis(t) / acilis(t)   - 1   -> bugun acilista hissedeysek kazanilir
Boylece alis gunu sadece gun ici, satis gunu sadece gece getirisi alinir.

Maliyetler her alis ve satista, islem acilista yapildigi icin gece ile gun ici
getirinin arasinda uygulanir:
  komisyon * (1 + BSMV) + kayma
Kayma (slippage): acilis fiyatindan tam alinamamasi, ornegin emrin karsi tarafa
gecmesi. Likit hisselerde kucuk, sig hisselerde buyuk olur.
"""
from dataclasses import dataclass

import pandas as pd

VARSAYILAN_SERMAYE = 100_000


@dataclass(frozen=True)
class Maliyet:
    """Islem basina maliyet oranlari (her alis ve her satis icin ayri ayri).

    Varsayilanlar bireysel yatirimci icin orta bir senaryodur (Kasim 2026):
    aracı kurum komisyonu hacme gore yaklasik on binde 9 - on binde 20 arasi, borsa payi
    dahil; BSMV komisyonun %5'i. Kayma bir varsayimdir: likit BIST 30 hissesinde
    acilista on binde 5. Kendi kurumunun oranini girmen daha dogru sonuc verir.
    """
    komisyon: float = 0.0015
    bsmv: float = 0.05
    kayma: float = 0.0005

    def __post_init__(self):
        for ad in ("komisyon", "bsmv", "kayma"):
            deger = getattr(self, ad)
            if not 0 <= deger < 1:
                raise ValueError(f"{ad} 0 ile 1 arasinda olmali, verilen: {deger}")

    @property
    def islem_basi(self):
        return self.komisyon * (1 + self.bsmv) + self.kayma


VARSAYILAN_MALIYET = Maliyet()
MALIYETSIZ = Maliyet(0, 0, 0)


@dataclass
class Sonuc:
    strateji: str
    getiri: pd.Series        # gunluk getiri
    sermaye: pd.Series       # gun sonu sermaye
    pozisyon: pd.Series      # gun icinde hissede miyiz (0/1)
    islemler: pd.DataFrame   # her alis-satis cifti
    baslangic_sermaye: float = VARSAYILAN_SERMAYE
    odenen_maliyet: float = 0.0   # TL

    @property
    def toplam_getiri(self):
        return self.sermaye.iloc[-1] / self.baslangic_sermaye - 1

    @property
    def islem_sayisi(self):
        return len(self.islemler)


def _kontrol(df):
    eksik = [s for s in ("acilis", "kapanis") if s not in df.columns]
    if eksik:
        raise ValueError(f"Veride eksik sutun: {', '.join(eksik)}")
    if len(df) < 2:
        raise ValueError("En az 2 gunluk veri gerekli")


def pozisyon_hesapla(sinyal):
    """t gunu kapanista verilen karar, t+1 gunu acilista uygulanir."""
    return sinyal.shift(1).fillna(0).astype("int8").rename("pozisyon")


def gunluk_getiri(df, pozisyon, maliyet=MALIYETSIZ):
    """(getiri, maliyet_orani) dondurur. Maliyet orani, o gun acilista odenen payin
    sermayeye orani (islem yoksa 0)."""
    onceki = pozisyon.shift(1).fillna(0)
    gece = (df["acilis"] / df["kapanis"].shift(1) - 1).fillna(0)
    gun_ici = df["kapanis"] / df["acilis"] - 1
    islem = (pozisyon - onceki).abs()
    maliyet_orani = islem * maliyet.islem_basi
    getiri = (1 + onceki * gece) * (1 - maliyet_orani) * (1 + pozisyon * gun_ici) - 1
    return getiri.rename("getiri"), maliyet_orani


def islem_listesi(df, pozisyon, maliyet=MALIYETSIZ):
    """Her alis (acilista) ve satis (acilista) icin bir satir. Sonuna kadar acik kalan
    islem son kapanistan degerlenir ve 'acik' olarak isaretlenir."""
    degisim = pozisyon.diff().fillna(pozisyon.iloc[0])
    alislar = list(pozisyon.index[degisim == 1])
    satislar = list(pozisyon.index[degisim == -1])
    satirlar = []
    for i, giris in enumerate(alislar):
        if i < len(satislar):
            cikis, cikis_fiyat, acik = satislar[i], df.at[satislar[i], "acilis"], False
        else:
            cikis, cikis_fiyat, acik = df.index[-1], df["kapanis"].iloc[-1], True
        giris_fiyat = df.at[giris, "acilis"]
        kesinti = (1 - maliyet.islem_basi) ** (1 if acik else 2)   # acik islemde satis maliyeti yok
        satirlar.append({"giris": giris, "giris_fiyat": giris_fiyat, "cikis": cikis,
                         "cikis_fiyat": cikis_fiyat, "getiri": cikis_fiyat / giris_fiyat * kesinti - 1,
                         "acik": acik})
    return pd.DataFrame(satirlar, columns=["giris", "giris_fiyat", "cikis", "cikis_fiyat", "getiri", "acik"])


def backtest(df, strateji, sermaye=VARSAYILAN_SERMAYE, maliyet=VARSAYILAN_MALIYET):
    _kontrol(df)
    if sermaye <= 0:
        raise ValueError(f"Sermaye pozitif olmali, verilen: {sermaye}")
    sinyal = strateji.sinyal(df)
    if not sinyal.index.equals(df.index):
        raise ValueError(f"{strateji} sinyali fiyat verisiyle ayni tarihleri kullanmiyor")
    pozisyon = pozisyon_hesapla(sinyal)
    getiri, maliyet_orani = gunluk_getiri(df, pozisyon, maliyet)
    seri = (sermaye * (1 + getiri).cumprod()).rename("sermaye")
    # Odenen maliyet = islem anindaki sermaye x maliyet orani
    onceki_sermaye = seri.shift(1).fillna(sermaye)
    gece = (df["acilis"] / df["kapanis"].shift(1) - 1).fillna(0)
    acilistaki = onceki_sermaye * (1 + pozisyon.shift(1).fillna(0) * gece)
    odenen = float((acilistaki * maliyet_orani).sum())
    return Sonuc(str(strateji), getiri, seri, pozisyon, islem_listesi(df, pozisyon, maliyet),
                 sermaye, odenen)


if __name__ == "__main__":
    import sys

    from src.strateji import AlTut, HareketliOrtalama
    from src.veri import yukle

    sembol = sys.argv[1] if len(sys.argv) > 1 else "THYAO.IS"
    df, _ = yukle(sembol, "2018-01-01", "2025-12-31")
    m = VARSAYILAN_MALIYET
    print(f"{sembol}, {df.index[0].date()} - {df.index[-1].date()}")
    print(f"Maliyet: komisyon on binde {m.komisyon * 1e4:g} + BSMV %{m.bsmv * 100:g} + kayma on binde "
          f"{m.kayma * 1e4:g} = islem basi %{m.islem_basi * 100:.3f}\n")
    print(f"{'Strateji':<38}{'Maliyetsiz':>11}{'Maliyetli':>11}{'Odenen TL':>11}{'Islem':>7}{'Kazanan':>9}")
    for s in (AlTut(), HareketliOrtalama(20, 50), HareketliOrtalama(50, 200)):
        ham = backtest(df, s, maliyet=MALIYETSIZ)
        r = backtest(df, s)
        kazanan = f"%{100 * (r.islemler['getiri'] > 0).mean():.0f}" if r.islem_sayisi else "-"
        print(f"{r.strateji:<38}{ham.toplam_getiri:>+11.0%}{r.toplam_getiri:>+11.0%}"
              f"{r.odenen_maliyet:>11,.0f}{r.islem_sayisi:>7}{kazanan:>9}")
