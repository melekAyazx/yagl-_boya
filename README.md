# Yagli Boya Atolyesi 8.0

Uygulama tamamen bilgisayarda calisir. API anahtari, API kredisi veya
donusturme sirasinda internet gerekmez. Fotograf bir servise gonderilmez.
Ucretli API modu ve OpenAI kutuphanesi bagimliligi kaldirilmistir.

## Kullanim

Onceki pencereyi kapatip `run_app.bat` dosyasini acin. Baslik 8.0 olmalidir.

1. `Fotograf ac` ile fotografi secin.
2. Varsayilan Referansli yagli boya stilini kullanin.
3. `Referans tablo sec` ile begendiginiz yagli boya resmini secin.
   Projedeki referans_yagli_boya_portre.png baslangic referansidir; asil
   istediginiz tabloyu secerek degistirebilirsiniz.
4. Firca boyutu, detay ve boya yogunlugunu duzenleyin. Varsayilan ayarlar
   referans stilini daha ince firca olceginde ve hafif kaynak karisimiyla
   uygular; boya dokusunu korurken sert firca izlerini yumusatir, doygunluk ve
   kontrasti yapay bicimde yukseltmez. `Fotograf
   renklerini koru` varsayilan olarak aciktir; kaynak fotografin renk ve
   parlaklik tonlarini buyuk olcude korur. Detay koruma yuzu ve ince cizgileri
   daha belirgin tutar. Daha stilize gorunum icin Boya yogunlugunu artirin;
   referans tablonun renklerini daha fazla aktarmak icin renk korumayi kapatin.
   Ust ve alt kisimda tekduze koyu bosluklar varsa donusturme sirasinda
   disarida birakilir; cikti boyutlari ve bu bosluklar korunur.
5. `Yagli boyaya donustur` dugmesine basin.
6. `Disa aktar` ile PNG, JPEG veya WebP kaydedin.

Referansli yagli boya, MicroAST sinir agini CPU'da calistirir. Bu model referansin
renk ve dokusunu fotograf yapisina aktarir. Model dosyalari bilgisayardadir;
API anahtari, kredi ve donusturme sirasinda internet gerekmez. Bu bir yerel stil
aktarimidir; sohbette uretilen portreyle ayni uretken model veya kalite garantisi degildir.
Sonucun kalitesi fotograf ve referansla birlikte degerlendirilmelidir.
768 pikselden buyuk fotograf kucultulerek islenip orijinal boyuta geri getirilir.
`local_reference_comparison.png` yerel modelin ornek fotograf karsilastirmasidir.
Portre, Dengeli ve Belirgin firca; katmanli raster firca darbelerini kullanir.
Klasik tablo, bilgisayardaki sabit Van Gogh stil aktarim modelini kullanir.
Kaynak ve MIT lisansi: https://github.com/EndyWon/MicroAST
Kaynak kodu ve model dosyalari vendor_microast klasorunde bulunur.

## Kurulum

Python 3.11+, tkinter ve requirements.txt kutuphaneleri gereklidir.
Bu bilgisayarda modeller ve kutuphaneler hazirdir.

```powershell
pip install -r requirements.txt
python main.py
python main.py input.jpg output.png
python main.py input.jpg output.png --style classical
python main.py input.jpg output.png --style reference --reference tablo.jpg
```

Windows CPU ortamini ilk kez kurarken PyTorch'u resmi CPU paketinden kurabilirsiniz:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Model dosyalarini uygulamayla ayni klasorde tutun:

- `face_detection_yunet_2023mar.onnx`: yerel yuz bulma.
- `haarcascade_frontalface_default.xml`: yedek yuz bulma.
- `starry_night.t7`: Klasik tablo modeli.

Model kaynaklari ve kullanim kosullari `MODEL-SOURCE.txt` ve
`LICENSE-YUNET.txt` dosyalarindadir. Sabit stil modelinin ticari kullanimi icin
model yazarindan izin alinmasi gerekir.

```powershell
python -m unittest test_app test_reference -v
```
