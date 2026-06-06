# Analiza emocji twarzy na zdjeciach

Srednio zaawansowany projekt rozpoznawania emocji twarzy na podstawie obrazow.
Projekt uzywa wlasnego modelu CNN trenowanego na danych FER-style w katalogach
`data/train` i `data/test`.

## Co zawiera projekt

- `model_definition.py` - definicja sredniego modelu CNN.
- `train.py` - trening modelu z validation split wydzielonym z `data/train`.
- `evaluate.py` - finalna ewaluacja na nietykanym `data/test`.
- `predict_image.py` - predykcja emocji na pojedynczym zdjeciu.
- `app.py` - aplikacja webowa FastAPI z kamera.
- `utils/helpers.py` - stale, preprocessing twarzy i funkcje pomocnicze.
- `templates/` i `static/` - frontend aplikacji.

## Model

Model pracuje na obrazach:

- grayscale,
- 48x48 px,
- 7 klas emocji:
  - angry,
  - disgust,
  - fear,
  - happy,
  - neutral,
  - sad,
  - surprise.

Architektura jest srednio zaawansowana:

- kilka blokow `Conv2D`,
- `BatchNormalization`,
- `MaxPooling2D`,
- `Dropout`,
- `GlobalAveragePooling2D`,
- klasyfikator `Dense`.

Nie ma tu transfer learningu ani duzego backbone typu EfficientNet/ResNet.
Dzieki temu trening jest krotszy i projekt jest prostszy do wyjasnienia.

## Instalacja

Aktywuj srodowisko:

```powershell
.\.venv\Scripts\activate
```

Zainstaluj zaleznosci, jesli trzeba:

```powershell
pip install -r requirements.txt
```

## Trening

```powershell
python train.py
```

Trening:

- uzywa `data/train`,
- automatycznie wydziela validation split,
- nie uzywa `data/test`,
- zapisuje najlepszy model do `models/best_model.keras`.

## Ewaluacja

```powershell
python evaluate.py
```

Ewaluacja uzywa tylko `data/test`.
Raporty zapisuja sie w `reports/`.

## Predykcja jednego zdjecia

```powershell
python predict_image.py --image "sciezka\do\zdjecia.jpg"
```

## Aplikacja webowa

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8002 --reload
```

Adres:

```text
http://127.0.0.1:8002
```

## Dodatkowa instrukcja

Dokladniejsza instrukcja uruchamiania jest w:

```text
INSTRUKCJA_MODELU.txt
```
