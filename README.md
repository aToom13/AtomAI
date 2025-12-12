# AtomAI

AtomAI, sıfırdan eğitilen ve ince ayar (fine-tuning) yapılan, matematik odaklı bir Dil Modeli (LLM) projesidir. Proje, kendi GPT mimarisini (`AtomAIBase`) kullanır ve PyTorch tabanlıdır.

## Proje Yapısı

Proje, işlevlerine göre modüler klasörlere ayrılmıştır:

### 1. `pretraining/` (Ön Eğitim)
Temel modelin (Base Model) eğitilmesi için gerekli dosyalar buradadır.
- **`download_data.py`**: OpenWebMath veri setini indirir ve hazırlar.
- **`data_loader.py`**: Veri yükleme ve inceleme yardımcı araçları.
- **`train.py`**: Temel modelin eğitimi (Pre-training) için ana script.

### 2. `sft/` (Supervised Fine-Tuning - SFT)
Temel modelin soru-cevap yeteneği kazanması için yapılan ince ayar işlemleri buradadır.
- **`prepare_sft_data.py`**: GSM8K veri setini indirir ve "Soru-Cevap" formatına dönüştürüp kaydeder.
- **`train_sft.py`**: Temel modeli (Base Model) alıp SFT verisiyle yeniden eğitir (Fine-Tuning).

### 3. `shared/` (Ortak Dosyalar)
Tüm modüller tarafından kullanılan ortak bileşenler.
- **`model.py`**: `AtomAIBase` model mimarisi (Transformer, Attention, MLP) ve `GPTConfig` yapılandırma sınıfı.

### 4. `inference/` (Kullanım / Çıkarım)
Eğitilmiş modelleri test etmek ve kullanmak için araçlar.
- **`chat.py`**: Model ile interaktif sohbet arayüzü (Terminal tabanlı).
- **`generate.py`**: Verilen bir başlangıç metnini (prompt) tamamlamak için script.

### `data/`
Eğitim verilerinin (`.txt`, `.bin`) kaydedildiği klasör. Git deposuna dahil edilmez (gitignore).

---

## Kullanım Kılavuzu

Scriptleri proje ana dizininden veya kendi klasörleri içinden çalıştırabilirsiniz.

### Adım 1: Ön Eğitim (Pre-training)

Önce veriyi indirin:
```bash
python pretraining/download_data.py
```

Modeli eğitin (Base Model):
```bash
python pretraining/train.py
```
*Bu işlem `atomai_base_model.pth` dosyasını oluşturur.*

### Adım 2: İnce Ayar (SFT)

SFT verisini hazırlayın (GSM8K):
```bash
python sft/prepare_sft_data.py
```

Modeli ince ayardan geçirin:
```bash
python sft/train_sft.py
```
*Bu işlem `atomai_sft_model.pth` dosyasını oluşturur.*

### Adım 3: Test ve Kullanım

Model ile sohbet edin:
```bash
python inference/chat.py --model_path atomai_sft_model.pth
```

Metin üretimi yapın:
```bash
python inference/generate.py --model_path atomai_base_model.pth --prompt "Theorem: "
```
