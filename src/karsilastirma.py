"""Stratejileri BIST 100 endeksi ve mevduat faiziyle karsilastirir.

Bir stratejinin "+%300" getirmesi tek basina bir sey soylemez. Sorulmasi gereken:
  1. Ayni donemde BIST 100'u alip tutmak ne getirdi? (piyasanin kendisi)
  2. Ayni parayi mevduata/faize koymak ne getirdi? (risk almadan ne kazanilirdi)
TL'de enflasyon yuksek oldugu icin 2. soru Turkiye'de ozellikle onemli.

Not: BIST 100 endeksinden 27 Temmuz 2020'de iki sifir atildi (endeks degeri 100'e
bolundu). Veri kaynagi bunu duzeltmemisse o gun %-99'luk sahte bir "cokus" gorunur;
olcek_duzelt() bu tur tek gunluk olcek degisimlerini bulup onceki veriyi olceklendirir.

Kullanim:
    python -m src.karsilastirma THYAO.IS --faiz 0.35
"""
import argparse

import numpy as np
import pandas as pd

from src.motor import backtest
from src.olcutler import maksimum_dusus, ozet, tablo, yillik_getiri
from src.strateji import AlTut, HareketliOrtalama

ENDEKS = "XU100.IS"
OLCEK_ESIGI = 20   # bir gunde 20 kattan fazla degisim gercek fiyat hareketi olamaz


def olcek_duzelt(df):
    """(df, uyarilar). Tek gunluk 20 kattan buyuk sicramalari olcek degisimi sayar ve
    o gunden onceki tum fiyatlari yeni olcege getirir. Hacim degistirilmez."""
    df = df.copy()
    uyarilar = []
    fiyatlar = [s for s in ("acilis", "yuksek", "dusuk", "kapanis") if s in df.columns]
    oran = df["kapanis"] / df["kapanis"].shift(1)
    for tarih in oran[(oran > OLCEK_ESIGI) | (oran < 1 / OLCEK_ESIGI)].index:
        k = df.at[tarih, "kapanis"] / df["kapanis"].shift(1).at[tarih]
        carpan = 10 ** round(np.log10(k))   # 0.01 -> 1/100
        df.loc[df.index < tarih, fiyatlar] *= carpan
        uyarilar.append(f"{tarih.date()}: olcek degisimi (x{carpan:g}) duzeltildi")
    return df, uyarilar


def hizala(*cerceveler):
    """Sadece hepsinde ortak olan islem gunlerini birakir."""
    ortak = cerceveler[0].index
    for c in cerceveler[1:]:
        ortak = ortak.intersection(c.index)
    if len(ortak) < 2:
        raise ValueError("Ortak islem gunu yok; tarih araliklarini kontrol et")
    return [c.loc[ortak] for c in cerceveler]


def faiz_egrisi(tarihler, yillik_faiz, sermaye=1.0):
    """Takvim zamanina gore (1+faiz)^(yil) buyuyen sermaye; yillik_getiri ile ayni
    zaman olcusu. Islem gunu sayisina gore buyutmek, yilda 252'den fazla gun olan
    veride faizi oldugundan yuksek gosteriyordu. Vergi/stopaj dahil degil."""
    yil = (tarihler - tarihler[0]).days / 365.25
    return pd.Series(sermaye * (1 + yillik_faiz) ** np.asarray(yil), index=tarihler, name="faiz")


def beta_korelasyon(getiri, endeks_getiri):
    """Stratejinin endeksle birlikte ne kadar hareket ettigi. beta 1: endeks kadar,
    0.5: yarisi kadar. Nakitte gecen gunler betayi dusurur."""
    g, e = getiri.align(endeks_getiri, join="inner")
    varyans = e.var(ddof=1)
    if not varyans or np.isnan(varyans):
        return None, None
    beta = float(g.cov(e, ddof=1) / varyans)
    korelasyon = float(g.corr(e)) if g.std(ddof=1) > 1e-12 else None   # hic islem yoksa tanimsiz
    return beta, korelasyon


def karsilastir(df, endeks, stratejiler, faiz=None, **backtest_kw):
    """Her strateji ve endeksin al-tut'u icin ozet; endeks ve faize gore fark eklenir."""
    df, endeks = hizala(df, endeks)
    endeks_sonuc = backtest(endeks, AlTut(), **backtest_kw)
    endeks_ozet = ozet(endeks_sonuc)
    endeks_ozet["strateji"] = "BIST 100 al-tut"
    faiz_yillik = None
    if faiz is not None:
        egri = faiz_egrisi(df.index, faiz)
        faiz_yillik = yillik_getiri(egri)

    sonuclar = []
    for s in stratejiler:
        r = backtest(df, s, **backtest_kw)
        o = ozet(r)
        o["beta"], o["korelasyon"] = beta_korelasyon(r.getiri, endeks_sonuc.getiri)
        o["endekse_gore"] = o["yillik_getiri"] - endeks_ozet["yillik_getiri"]
        o["faize_gore"] = None if faiz_yillik is None else o["yillik_getiri"] - faiz_yillik
        sonuclar.append(o)
    endeks_ozet.update({"beta": 1.0, "korelasyon": 1.0, "endekse_gore": 0.0,
                        "faize_gore": None if faiz_yillik is None else endeks_ozet["yillik_getiri"] - faiz_yillik})
    return sonuclar + [endeks_ozet], faiz_yillik


def fark_tablosu(ozetler):
    def y(x):
        return "-" if x is None else f"{x:+.1%}"

    s = [f"{'Strateji':<38}{'Yillik':>9}{'Endekse gore':>14}{'Faize gore':>12}{'Beta':>7}{'Korel.':>8}",
         "-" * 88]
    for o in ozetler:
        beta = "-" if o["beta"] is None else f"{o['beta']:.2f}"
        kor = "-" if o["korelasyon"] is None else f"{o['korelasyon']:.2f}"
        s.append(f"{o['strateji'][:37]:<38}{y(o['yillik_getiri']):>9}{y(o['endekse_gore']):>14}"
                 f"{y(o['faize_gore']):>12}{beta:>7}{kor:>8}")
    return "\n".join(s)


if __name__ == "__main__":
    from src.veri import yukle

    ap = argparse.ArgumentParser()
    ap.add_argument("sembol", nargs="?", default="THYAO.IS")
    ap.add_argument("--baslangic", default="2018-01-01")
    ap.add_argument("--bitis", default="2025-12-31")
    ap.add_argument("--faiz", type=float, default=None,
                    help="Donemin ortalama yillik mevduat faizi, or. 0.35 (kendi kaynagindan gir)")
    a = ap.parse_args()

    df, _ = yukle(a.sembol, a.baslangic, a.bitis)
    endeks, uyarilar = yukle(ENDEKS, a.baslangic, a.bitis)
    endeks, olcek = olcek_duzelt(endeks)
    for u in olcek:
        print("uyari:", u)

    stratejiler = [AlTut(), HareketliOrtalama(20, 50), HareketliOrtalama(50, 200)]
    ozetler, faiz_yillik = karsilastir(df, endeks, stratejiler, faiz=a.faiz)
    print(f"\n{a.sembol} ve {ENDEKS}, {df.index[0].date()} - {df.index[-1].date()}, maliyetli\n")
    print(tablo(ozetler))
    print()
    print(fark_tablosu(ozetler))
    if faiz_yillik is None:
        print("\nFaiz karsilastirmasi icin --faiz ile donemin ortalama mevduat faizini gir (or. --faiz 0.30).")
    md = maksimum_dusus(backtest(*hizala(endeks), AlTut()).sermaye)
    print(f"\nBIST 100'un en buyuk dususu: %{md['oran'] * 100:.0f} ({md['zirve'].date()} -> {md['dip'].date()})")
