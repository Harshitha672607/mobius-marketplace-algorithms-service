from fastapi import APIRouter, HTTPException
from app.services.valuation_service import ValuationService

router = APIRouter(prefix="/valuation", tags=["Valuation"])

service = ValuationService()

@router.get("/{product_id}")
def calculate_valuation(product_id: str):
    value = service.get_valuation(product_id)
    if value is None:
        raise HTTPException(status_code=404, detail="Product not found")
    
    return {
        "productId": product_id,
        "estimatedValue": value
    }
