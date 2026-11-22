"""Backtest motoru: stratejinin sinyallerini gunluk getiriye ve sermaye egrisine cevirir.

Zamanlama (en kolay hata yapilan yer):
  - Sinyal t gunu KAPANISTA hesaplanir (o gunun kapanis fiyati ancak o zaman bilinir).
  - Islem t+1 gunu ACILISTA yapilir. t gununun kapanisindan islem yapmak, o fiyati
    onceden bilmek demektir: bu bir lookahead hatasidir ve sonuclari iyi gosterir.

Gunluk getiri iki parcaya bolunur:
  gece    = acilis(t) / kapanis(t-1) - 1   -> dun aksam hissedeysek kazanilir
  gun ici = kapanis(t) / acilis(t)   - 1   -> bugun acilista hissedeysek kazanilir
Boylece alis gunu sadece gun ici, satis gunu sadece gece getirisi alinir.

Maliyetler (komisyon, BSMV, kayma) Gun 4'te eklenecek; simdilik sonuclar maliyetsiz.
"""
from dataclasses import dataclass

import pandas as pd

VARSAYILAN_SERMAYE = 100_000


@dataclass
class Sonuc:
    strateji: str
    getiri: pd.Series        # gunluk getiri
    sermaye: pd.Series       # gun sonu sermaye
    pozisyon: pd.Series      # gun icinde hissede miyiz (0/1)
    islemler: pd.DataFrame   # her alis-satis cifti
    baslangic_sermaye: float = VARSAYILAN_SERMAYE

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


def gunluk_getiri(df, pozisyon):
    onceki = pozisyon.shift(1).fillna(0)
    gece = (df["acilis"] / df["kapanis"].shift(1) - 1).fillna(0)
    gun_ici = df["kapanis"] / df["acilis"] - 1
    return ((1 + onceki * gece) * (1 + pozisyon * gun_ici) - 1).rename("getiri")


def islem_listesi(df, pozisyon):
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
        satirlar.append({"giris": giris, "giris_fiyat": giris_fiyat, "cikis": cikis,
                         "cikis_fiyat": cikis_fiyat, "getiri": cikis_fiyat / giris_fiyat - 1,
                         "acik": acik})
    return pd.DataFrame(satirlar, columns=["giris", "giris_fiyat", "cikis", "cikis_fiyat", "getiri", "acik"])


def backtest(df, strateji, sermaye=VARSAYILAN_SERMAYE):
    _kontrol(df)
    if sermaye <= 0:
        raise ValueError(f"Sermaye pozitif olmali, verilen: {sermaye}")
    sinyal = strateji.sinyal(df)
    if not sinyal.index.equals(df.index):
        raise ValueError(f"{strateji} sinyali fiyat verisiyle ayni tarihleri kullanmiyor")
    pozisyon = pozisyon_hesapla(sinyal)
    getiri = gunluk_getiri(df, pozisyon)
    return Sonuc(str(strateji), getiri, (sermaye * (1 + getiri).cumprod()).rename("sermaye"),
                 pozisyon, islem_listesi(df, pozisyon), sermaye)


if __name__ == "__main__":
    import sys

    from src.strateji import AlTut, HareketliOrtalama
    from src.veri import yukle

    sembol = sys.argv[1] if len(sys.argv) > 1 else "THYAO.IS"
    df, _ = yukle(sembol, "2018-01-01", "2025-12-31")
    print(f"{sembol}, {df.index[0].date()} - {df.index[-1].date()}, MALIYETSIZ (komisyon Gun 4'te)\n")
    print(f"{'Strateji':<38}{'Toplam getiri':>14}{'Son sermaye':>14}{'Islem':>7}{'Kazanan':>9}")
    for s in (AlTut(), HareketliOrtalama(20, 50), HareketliOrtalama(50, 200)):
        r = backtest(df, s)
        kazanan = f"%{100 * (r.islemler['getiri'] > 0).mean():.0f}" if r.islem_sayisi else "-"
        print(f"{r.strateji:<38}{r.toplam_getiri:>+14.0%}{r.sermaye.iloc[-1]:>14,.0f}{r.islem_sayisi:>7}{kazanan:>9}")
