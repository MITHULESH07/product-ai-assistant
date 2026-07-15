from pydantic import BaseModel


class ProductInfo(BaseModel):
    id: str
    name: str


class ProductListResponse(BaseModel):
    products: list[ProductInfo]
