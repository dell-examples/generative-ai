# Created by Scalers AI for Dell Inc.
import os

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()
ip_address = os.environ.get("IP_ADDRESS", "localhost")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="./")


@app.get("/")
def ui(request: Request):
    data = {"ip_address": ip_address}
    return templates.TemplateResponse(
        "index.html", context={"request": request, "data": data}
    )
