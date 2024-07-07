from fastapi import FastAPI, Request, Body, Form
from fastapi.responses import HTMLResponse
import logging
from typing import Optional, Any, Dict
from pydantic import BaseModel, Field

app = FastAPI(
    title='Quotes by Nicolacus Maximus The Third',
    description="Get a real quote said by Nicolacus Maximus the Third himself",
    servers=[
        {   # Make sure you don't have a slash at the end of the url
            "url": "https://dance-increased-julian-slots.trycloudflare.com",
        }
    ]
)


class Quote(BaseModel):
    quote: str = Field(
        description="The quote that Nicolas Maximus said",
    )
    year: Optional[int] = Field(
        description="The year when Nicolacus Maximus said the quote.",
    )


# Configure logging
logging.basicConfig(level=logging.INFO)


@app.get(
    "/quote",
    summary="Return a random quote by Nicolacus Maximus",
    description="Upon receiving a GET request this endpoint will return a real quote said by Nicolacus Maximus himself.",
    response_description="A Quote object that contains the quote said by Nicolacus Maximus and the date when the quote was said.",
    response_model=Quote,
    openapi_extra={
        "x-openai-isConsequential": False,
    }
)
def get_quote(request: Request):
    logging.info(f"Endpoint {request.url.path} was called")
    print(request.headers)
    return {"quote": "Life is short so enjoy whatever you can eat", "year": 1950}


user_token_db = {
    "ABCDE": "minho"
}


@app.get("/authorize",
         response_class=HTMLResponse,
         include_in_schema=False,
         )
def handle_authorize(client_id: str, redirect_uri: str, state: str):
    return f"""
    <html>
        <head>
            <title>Nicolacus Maximus Log In</title>
        </head>
        <body>
            <h1> Log into Nicolacus Maximus</h1>
            <a href="{redirect_uri}?code=ABCDE&state={state}">Authorize NicolacusMaximus GPT</a>
        </body>
    </html>
    """


@app.post("/token",
          include_in_schema=False,)
def handle_token(code=Form(...)):
    return {
        "access_token": user_token_db[code]
    }
