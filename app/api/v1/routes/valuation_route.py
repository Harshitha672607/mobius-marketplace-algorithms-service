from fastapi import APIRouter, HTTPException
from app.agents.valuation_engine import (
    ValuationEngine,
    MockPILayer,
    BondingCurveParams,
    ValuationInputFactor,
    ReputationSnapshot,
    StakingConfig,
    RewardPolicyParams,
    DemandSignal,
    ProductValuation
)
from app.models.valuation_models import Alg6Input,Alg7Input,AlgResponse,AlgorithmConfig
router = APIRouter(prefix="/valuation", tags=["Valuation Algorithms"])

# initialize engine + mock DB
mock_pi = MockPILayer()
engine = ValuationEngine(mock_pi)


# --------------------------- ALG-1 MAIN ENGINE ---------------------------
@router.get("/{product_node_id}")
def run_alg1_master(product_node_id: str):
    result = engine.calculate_product_value(product_node_id)
    if result is None:
        raise HTTPException(404, f"Product {product_node_id} not found")
    return {"product_node_id": product_node_id, "estimated_value": result}


# --------------------------- ALG-2: Bonding Curve ---------------------------
@router.post("/alg2/bonding-curve")
def run_alg2(params: BondingCurveParams):
    price = engine._calculate_bonding_curve_price(params)
    return {"algorithm": "ALG-2 Bonding Curve", "price": price}


# --------------------------- ALG-3: Multi-Factor ---------------------------
@router.post("/alg3/multi-factor")
def run_alg3(factors: list[ValuationInputFactor]):
    fake_product = ProductValuation(
        productNodeId="temp",
        algorithm={"algType": "MARKET_COMPARABLE"},
        input_factors=factors
    )
    price = engine._estimate_multi_factor_value(fake_product)
    return {"algorithm": "ALG-3 Multi-Factor", "price": price}


# --------------------------- ALG-4: Reputation Weighted ---------------------------
@router.post("/alg4/reputation")
def run_alg4(neutral_price: float, reputation: ReputationSnapshot):
    price = engine._calculate_reputation_weighted_value(neutral_price, reputation)
    return {"algorithm": "ALG-4 Reputation", "price": price}


# --------------------------- ALG-5: Dynamic Bonding Curve ---------------------------
@router.post("/alg5/dynamic-curve")
def run_alg5(base_price: float, demand: DemandSignal, curve: BondingCurveParams):
    fake_product = ProductValuation(
        productNodeId="temp",
        algorithm={"algType": "BONDING_CURVE", "bondingCurve": curve},
        demandSignal=demand
    )
    final_price = engine._calculate_dynamic_bonding_curve(base_price, fake_product)
    return {"algorithm": "ALG-5 Dynamic Curve", "adjusted_price": final_price}


# --------------------------- ALG-6: Yield Factor ---------------------------
# @router.post("/alg6/yield-factor")
# def run_alg6(staking: StakingConfig, reward: RewardPolicyParams, behavior_score: float):
#     factor = engine._model_token_yield_as_factor(staking, reward, behavior_score)
#     return {"algorithm": "ALG-6 Yield Factor", "yield_factor": factor}
@router.post("/alg6/yield-factor") 
def run_alg6(input_data: Alg6Input): 
    factor = engine._model_token_yield_as_factor( input_data.staking, input_data.reward, input_data.behavior_score ) 
    return {"algorithm": "ALG-6 Yield Factor", "yield_factor": factor}

# --------------------------- ALG-7: Hybrid Composite ---------------------------
@router.post("/alg7/hybrid")
def run_alg7(product: ProductValuation):
    price = engine._calculate_hybrid_value(product)
    return {"algorithm": "ALG-7 Hybrid Composite", "price": price}
