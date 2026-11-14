from fastapi import FastAPI
from .router import router

def main():
    app = FastAPI(
    title="Text Summarization API",
    description="An API for text summarization using different models.",
    version="1.0.0",
    contact={
        "name": "Adeola Wuraola Aderibigbe",
        "email": "adeolaaderibigbe09@gmail.com"
    }
)

    app.include_router(router)

    return app

if __name__ == "__main__":
    import uvicorn

    app = main()
    
    uvicorn.run(app, host="127.0.0.1", port=8000)
