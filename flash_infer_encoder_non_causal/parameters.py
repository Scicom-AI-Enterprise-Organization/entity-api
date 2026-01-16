from pydantic import BaseModel, Field
from typing import List, Optional, Union


class EntityRequest(BaseModel):
    """Request for entity extraction/classification"""
    texts: List[str] = Field(default=["Hello world"], description="List of texts to process")
    max_length: Optional[int] = Field(default=512, description="Maximum sequence length")
    

class EntityResponse(BaseModel):
    """Response for entity extraction"""
    id: str
    results: List[dict]
    usage: dict


class BatchEntityRequest(BaseModel):
    """Batch request for entity processing"""
    inputs: List[str] = Field(default=["Hello world"], description="List of input texts")
    batch_size: Optional[int] = Field(default=None, description="Override batch size")


class TokenClassificationRequest(BaseModel):
    """Request for token classification"""
    text: Union[str, List[str]] = Field(default="Hello world", description="Text or list of texts")
    return_tensors: bool = Field(default=False, description="Return raw tensor outputs")
