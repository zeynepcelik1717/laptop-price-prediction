from pathlib import Path
import pickle

import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "laptop_price_model.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_names.pkl"
DATA_PATH = BASE_DIR / "data" / "laptop_price.csv"

app = FastAPI(title="Laptop Price Prediction")

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

with open(FEATURE_PATH, "rb") as file:
    feature_names = list(pickle.load(file))

    raw_df = pd.read_csv(DATA_PATH, encoding="latin-1")

feature_index = {name: i for i, name in enumerate(feature_names)}


def get_options(prefix):
    values = []

    for col in feature_names:
        if col.startswith(prefix + "_"):
            values.append(col.replace(prefix + "_", ""))

    return sorted(values)


options = {
    "companies": sorted(raw_df["Company"].unique().tolist())
}


def set_feature(row, feature_name, value):
    if feature_name in feature_index:
        row[feature_index[feature_name]] = value
    else:
        print("Feature bulunamadı:", feature_name)

@app.get("/options/{company}")
def get_company_options(company: str):

    filtered = raw_df[raw_df["Company"] == company]

    return {
        "products": sorted(filtered["Product"].unique().tolist()),
        "typenames": sorted(filtered["TypeName"].unique().tolist()),
        "resolutions": sorted(filtered["ScreenResolution"].unique().tolist()),
        "cpus": sorted(filtered["Cpu"].unique().tolist()),
        "memories": sorted(filtered["Memory"].unique().tolist()),
        "gpus": sorted(filtered["Gpu"].unique().tolist()),
        "opsys": sorted(filtered["OpSys"].unique().tolist())
    }

@app.get("/options/{company}/{product}")
def get_product_options(company: str, product: str):

    filtered = raw_df[
        (raw_df["Company"] == company) &
        (raw_df["Product"] == product)
    ]

    return {
        "typenames": sorted(filtered["TypeName"].unique().tolist()),
        "resolutions": sorted(filtered["ScreenResolution"].unique().tolist()),
        "cpus": sorted(filtered["Cpu"].unique().tolist()),
        "memories": sorted(filtered["Memory"].unique().tolist()),
        "gpus": sorted(filtered["Gpu"].unique().tolist()),
        "opsys": sorted(filtered["OpSys"].unique().tolist())
    }


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "options": options,
            "prediction": None,
            "error": None
        }
    )


@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    Company: str = Form(...),
    Product: str = Form(...),
    TypeName: str = Form(...),
    Inches: float = Form(...),
    ScreenResolution: str = Form(...),
    Cpu: str = Form(...),
    Ram: int = Form(...),
    Memory: str = Form(...),
    Gpu: str = Form(...),
    OpSys: str = Form(...),
    Weight: float = Form(...)
):
    try:
        row = [0.0] * len(feature_names)

        set_feature(row, "Inches", float(Inches))
        set_feature(row, "Ram", float(Ram))
        set_feature(row, "Weight", float(Weight))

        set_feature(row, f"Company_{Company}", 1.0)
        set_feature(row, f"Product_{Product}", 1.0)
        set_feature(row, f"TypeName_{TypeName}", 1.0)
        set_feature(row, f"ScreenResolution_{ScreenResolution}", 1.0)
        set_feature(row, f"Cpu_{Cpu}", 1.0)
        set_feature(row, f"Memory_{Memory}", 1.0)
        set_feature(row, f"Gpu_{Gpu}", 1.0)
        set_feature(row, f"OpSys_{OpSys}", 1.0)

        input_data = pd.DataFrame([row], columns=feature_names)

        prediction = model.predict(input_data)[0]

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "options": options,
                "prediction": round(float(prediction), 2),
                "error": None
            }
        )

    except Exception as e:
        print("TAHMIN HATASI:", e)

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "options": options,
                "prediction": None,
                "error": str(e)
            }
        )