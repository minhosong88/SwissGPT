from fastapi import FastAPI, Request
import logging
from typing import Optional
from pydantic import BaseModel, Field

app = FastAPI(
    title='Quotes by Nicolacus Maximus The Third',
    description="Get a real quote said by Nicolacus Maximus the Third himself",
    servers=[
        {   # Make sure you don't have a slash at the end of the url
            "url": "https://blackberry-remain-golden-non.trycloudflare.com",
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
    return {"quote": "Life is short so enjoy whatever you can eat", "year": 1950}
