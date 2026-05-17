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


feature_index = {name: i for i, name in enumerate(feature_names)}

raw_df = pd.read_csv(DATA_PATH, encoding="latin-1")

options = {
    "companies": sorted(raw_df["Company"].dropna().unique().tolist())
}


def set_feature(row, feature_name, value):
    if feature_name in feature_index:
        row[feature_index[feature_name]] = value
    else:
        print("Feature bulunamadı:", feature_name)


def create_selected_data(
    Company,
    Product,
    TypeName,
    Inches,
    ScreenResolution,
    Cpu,
    Ram,
    Memory,
    Gpu,
    OpSys,
    Weight
):
    return {
        "Company": Company,
        "Product": Product,
        "TypeName": TypeName,
        "Inches": Inches,
        "ScreenResolution": ScreenResolution,
        "Cpu": Cpu,
        "Ram": Ram,
        "Memory": Memory,
        "Gpu": Gpu,
        "OpSys": OpSys,
        "Weight": Weight
    }


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "options": options,
            "prediction": None,
            "error": None,
            "selected": None
        }
    )


@app.get("/options/{company}")
def get_company_options(company: str):
    filtered = raw_df[raw_df["Company"] == company]

    return {
        "products": sorted(filtered["Product"].dropna().unique().tolist())
    }


@app.get("/options/{company}/{product}")
def get_product_options(company: str, product: str):
    filtered = raw_df[
        (raw_df["Company"] == company) &
        (raw_df["Product"] == product)
    ]

    return {
        "typenames": sorted(filtered["TypeName"].dropna().unique().tolist()),
        "resolutions": sorted(filtered["ScreenResolution"].dropna().unique().tolist()),
        "cpus": sorted(filtered["Cpu"].dropna().unique().tolist()),
        "memories": sorted(filtered["Memory"].dropna().unique().tolist()),
        "gpus": sorted(filtered["Gpu"].dropna().unique().tolist()),
        "opsys": sorted(filtered["OpSys"].dropna().unique().tolist())
    }


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
    selected = create_selected_data(
        Company,
        Product,
        TypeName,
        Inches,
        ScreenResolution,
        Cpu,
        Ram,
        Memory,
        Gpu,
        OpSys,
        Weight
    )

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
                "error": None,
                "selected": selected
            }
        )

    except Exception as error:
        print("TAHMIN HATASI:", error)

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "options": options,
                "prediction": None,
                "error": str(error),
                "selected": selected
            }
        )