from typing import Annotated

from fastapi import APIRouter
from fastapi import Query, Depends
from .services.auth import verify_api_key

from .model.type import ModelParams

router = APIRouter(
    tags=["Text summarization"],
    responses={404: {"description": "Not found"}},
)

@router.get("/summarize", summary="Summarize text")
async def summarize(text: str, model: Annotated[ModelParams, Query()], api_key: str = Depends(verify_api_key)):
    """
    Summarize the provided text using the specified model.

    Args:
        text (str): The text to be summarized.
        model (ModelParams): The model parameters, including the model name.

    Returns:
        dict: A dictionary containing the summarized text.
    """
    if not verify_api_key(api_key):
        return {"error": "Invalid API key."}
    
    if model.name == "t5":
        from .predict.t5_inference import predict
    elif model.name == "qwen":
        from .predict.qwen_inference import predict
    else:
        raise ValueError("Invalid model name. Must be either 't5' or 'qwen'.")

    summary = predict(text)
    return {"summary": summary}