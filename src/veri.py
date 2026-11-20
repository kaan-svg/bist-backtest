"""Fiyat verisini indirir, onbellege alir ve temizler.

Veri kaynagi Yahoo Finance (yfinance). BIST hisseleri '.IS' ekiyle yazilir:
THYAO.IS, ASELS.IS; BIST 100 endeksi XU100.IS.

Fiyatlar temettu ve bolunmelere gore duzeltilmis (auto_adjust) gelir; boylece
bedelsiz sermaye artirimi gunlerinde sahte bir "cokus" gorulmez.

Kullanim:
    python -m src.veri THYAO.IS 2018-01-01 2025-12-31
"""
from pathlib import Path

import pandas as pd

VERI = Path(__file__).resolve().parent.parent / "data"
SUTUNLAR = ["acilis", "yuksek", "dusuk", "kapanis", "hacim"]
_YAHOO = {"Open": "acilis", "High": "yuksek", "Low": "dusuk", "Close": "kapanis", "Volume": "hacim"}
BOSLUK_UYARI_GUN = 7   # bu kadar takvim gunu veri yoksa uyar (bayram tatilleri ~4-5 gun)


def _yahoo_indir(sembol, baslangic, bitis):
    import yfinance as yf
    return yf.download(sembol, start=baslangic, end=bitis, auto_adjust=True,
                       progress=False, multi_level_index=False)


def _bicimle(ham):
    """yfinance ciktisini standart sutunlara cevirir. Eski/yeni surumlerdeki
    cok seviyeli sutun basliklarini da duzlestirir."""
    df = ham.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    eksik = [s for s in _YAHOO if s not in df.columns]
    if eksik:
        raise ValueError(f"Veride eksik sutun: {', '.join(eksik)}")
    df = df[list(_YAHOO)].rename(columns=_YAHOO)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "tarih"
    return df


def temizle(df):
    """(temiz_df, uyarilar) dondurur.

    - tarihe gore siralar, tekrar eden gunlerden sonuncusunu tutar
    - kapanisi bos veya sifir/negatif olan gunleri atar
    - yuksek < dusuk gibi tutarsiz satirlari atar
    - tatil dolgularini atar: Yahoo borsanin kapali oldugu gunler (bayram, yilbasi)
      icin hacmi 0, acilis=yuksek=dusuk=kapanis olan satirlar uretiyor. Bunlar kalirsa
      backtest bu gunleri "fiyati oynamayan islem gunu" sayar: volatilite dusuk,
      Sharpe yuksek cikar ve strateji borsanin kapali oldugu gun islem yapabilir.
    - uzun veri bosluklarini uyari olarak raporlar (atmaz)
    """
    uyarilar = []
    df = df.sort_index()
    tekrar = df.index.duplicated(keep="last")
    if tekrar.any():
        uyarilar.append(f"{tekrar.sum()} tekrar eden gun silindi")
        df = df[~tekrar]

    gecersiz = df["kapanis"].isna() | (df["kapanis"] <= 0)
    if gecersiz.any():
        uyarilar.append(f"{gecersiz.sum()} gun bos/gecersiz kapanis nedeniyle silindi")
        df = df[~gecersiz]

    tutarsiz = df["yuksek"] < df["dusuk"]
    if tutarsiz.any():
        uyarilar.append(f"{tutarsiz.sum()} gun yuksek < dusuk oldugu icin silindi")
        df = df[~tutarsiz]

    dolgu = ((df["hacim"] == 0) & (df["acilis"] == df["kapanis"])
             & (df["yuksek"] == df["kapanis"]) & (df["dusuk"] == df["kapanis"]))
    if dolgu.any():
        uyarilar.append(f"{dolgu.sum()} tatil dolgusu gun silindi (hacim 0, fiyat sabit)")
        df = df[~dolgu]

    if len(df) > 1:
        araliklar = df.index.to_series().diff().dt.days
        for tarih, gun in araliklar[araliklar > BOSLUK_UYARI_GUN].items():
            uyarilar.append(f"{tarih.date()} oncesinde {int(gun)} gunluk veri boslugu")

    if df.empty:
        raise ValueError("Temizlikten sonra veri kalmadi")
    return df, uyarilar


def onbellek_yolu(sembol, baslangic, bitis, klasor=VERI):
    return Path(klasor) / f"{sembol.replace('^', '')}_{baslangic}_{bitis}.csv"


def yukle(sembol, baslangic, bitis, indirici=_yahoo_indir, klasor=VERI, yenile=False):
    """Veriyi onbellekten okur, yoksa indirip kaydeder. (df, uyarilar) dondurur.

    indirici: test icin sahte bir fonksiyon verilebilir (internet gerekmez).
    """
    yol = onbellek_yolu(sembol, baslangic, bitis, klasor)
    if yol.exists() and not yenile:
        df = pd.read_csv(yol, index_col="tarih", parse_dates=["tarih"])
    else:
        ham = indirici(sembol, baslangic, bitis)
        if ham is None or len(ham) == 0:
            raise ValueError(f"{sembol} icin veri gelmedi. Sembol dogru mu (BIST icin '.IS' eki: THYAO.IS)? "
                             "Internet baglantisi veya Yahoo erisimi de sorun olabilir.")
        df = _bicimle(ham)
        yol.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(yol)
    return temizle(df[SUTUNLAR])


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        raise SystemExit("Kullanim: python -m src.veri SEMBOL BASLANGIC BITIS  (or. THYAO.IS 2018-01-01 2025-12-31)")
    df, uyarilar = yukle(*sys.argv[1:])
    print(df.tail())
    print(f"\n{len(df)} gun, {df.index[0].date()} - {df.index[-1].date()}")
    for u in uyarilar:
        print("uyari:", u)
