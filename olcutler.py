"""Performans olcutleri: bir backtest sonucunu karsilastirilabilir sayilara cevirir.

Toplam getiri tek basina yaniltir: ayni getiriye %20 dususle de %60 dususle de
ulasilabilir. Bu yuzden getiri, risk (volatilite, maksimum dusus) ve islem
istatistikleri birlikte raporlanir.

Turkiye icin not: TL getirileri yuksek enflasyon nedeniyle nominal olarak buyuk
gorunur. Sharpe orani risksiz faiz 0 kabul edilerek hesaplanirsa cok iyimser cikar;
karsilastirma icin risksiz_yillik parametresine o donemin mevduat/politika faizini
girmek gerekir. Strateji ile al-tut'u karsilastirirken ikisi ayni sekilde etkilendigi
icin siralama degismez.
"""
import math

import pandas as pd

ISLEM_GUNU = 252


def yillik_getiri(sermaye):
    """Bilesik yillik getiri (CAGR), takvim gunune gore."""
    yil = (sermaye.index[-1] - sermaye.index[0]).days / 365.25
    if yil <= 0:
        raise ValueError("Yillik getiri icin en az iki farkli tarih gerekli")
    return (sermaye.iloc[-1] / sermaye.iloc[0]) ** (1 / yil) - 1


def yillik_volatilite(getiri):
    return float(getiri.std(ddof=1) * math.sqrt(ISLEM_GUNU))


def sharpe(getiri, risksiz_yillik=0.0):
    """(ortalama gunluk fazla getiri / gunluk std) * sqrt(252). Volatilite 0 ise None."""
    gunluk_risksiz = (1 + risksiz_yillik) ** (1 / ISLEM_GUNU) - 1
    fazla = getiri - gunluk_risksiz
    std = fazla.std(ddof=1)
    if math.isnan(std) or std < 1e-12:   # sabit getiri: kayan nokta std'si tam 0 cikmaz
        return None
    return float(fazla.mean() / std * math.sqrt(ISLEM_GUNU))


def dusus_serisi(sermaye):
    """Her gun, o gune kadarki zirveden ne kadar asagida olundugu (0 veya negatif)."""
    return (sermaye / sermaye.cummax() - 1).rename("dusus")


def maksimum_dusus(sermaye):
    """{'oran', 'zirve', 'dip', 'toparlanma'}. Toparlanma: zirve seviyesine geri
    donulen ilk gun; donulmediyse None."""
    d = dusus_serisi(sermaye)
    dip = d.idxmin()
    zirve = sermaye.loc[:dip].idxmax()
    sonrasi = sermaye.loc[dip:]
    geri = sonrasi[sonrasi >= sermaye.loc[zirve]]
    return {"oran": float(d.min()), "zirve": zirve, "dip": dip,
            "toparlanma": geri.index[0] if len(geri) else None}


def ozet(sonuc, risksiz_yillik=0.0):
    """Backtest sonucunun tum olcutleri tek sozlukte."""
    md = maksimum_dusus(sonuc.sermaye)
    islemler = sonuc.islemler
    return {
        "strateji": sonuc.strateji,
        "toplam_getiri": sonuc.toplam_getiri,
        "yillik_getiri": yillik_getiri(pd.concat([
            pd.Series([sonuc.baslangic_sermaye], index=[sonuc.sermaye.index[0]]), sonuc.sermaye.iloc[1:]])),
        "yillik_volatilite": yillik_volatilite(sonuc.getiri),
        "sharpe": sharpe(sonuc.getiri, risksiz_yillik),
        "maksimum_dusus": md["oran"],
        "dusus_zirve": md["zirve"], "dusus_dip": md["dip"], "toparlanma": md["toparlanma"],
        "piyasada": float(sonuc.pozisyon.mean()),
        "islem_sayisi": len(islemler),
        "kazanan_orani": float((islemler["getiri"] > 0).mean()) if len(islemler) else None,
        "ortalama_islem": float(islemler["getiri"].mean()) if len(islemler) else None,
        "odenen_maliyet": sonuc.odenen_maliyet,
    }


def tablo(ozetler):
    """Birden fazla stratejinin ozetini terminalde okunur tabloya cevirir."""
    def yuzde(x, isaret=False):
        return "-" if x is None else (f"{x:+.0%}" if isaret else f"{x:.0%}")

    baslik = (f"{'Strateji':<38}{'Toplam':>8}{'Yillik':>8}{'Vol':>6}{'Sharpe':>8}"
              f"{'MaksDusus':>10}{'Piyasada':>9}{'Islem':>6}{'Kazanan':>8}")
    satirlar = [baslik, "-" * len(baslik)]
    for o in ozetler:
        sh = "-" if o["sharpe"] is None else f"{o['sharpe']:.2f}"
        satirlar.append(
            f"{o['strateji'][:37]:<38}{yuzde(o['toplam_getiri'], True):>8}{yuzde(o['yillik_getiri'], True):>8}"
            f"{yuzde(o['yillik_volatilite']):>6}{sh:>8}{yuzde(o['maksimum_dusus']):>10}"
            f"{yuzde(o['piyasada']):>9}{o['islem_sayisi']:>6}{yuzde(o['kazanan_orani']):>8}")
    return "\n".join(satirlar)


if __name__ == "__main__":
    import sys

    from src.motor import backtest
    from src.strateji import AlTut, HareketliOrtalama
    from src.veri import yukle

    sembol = sys.argv[1] if len(sys.argv) > 1 else "THYAO.IS"
    df, _ = yukle(sembol, "2018-01-01", "2025-12-31")
    sonuclar = [ozet(backtest(df, s)) for s in (AlTut(), HareketliOrtalama(20, 50), HareketliOrtalama(50, 200))]
    print(f"{sembol}, {df.index[0].date()} - {df.index[-1].date()}, maliyetli, risksiz faiz 0 (nominal TL)\n")
    print(tablo(sonuclar))
    for o in sonuclar:
        t = o["toparlanma"].date() if o["toparlanma"] is not None else "henuz toparlanmadi"
        print(f"\n{o['strateji']}: en buyuk dusus {o['dusus_zirve'].date()} -> {o['dusus_dip'].date()}, "
              f"toparlanma: {t}")
