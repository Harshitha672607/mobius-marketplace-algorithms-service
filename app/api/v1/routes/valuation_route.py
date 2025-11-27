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
from app.models.valuation_models import ValuationAlgorithm

from app.models.valuation_models import Alg6Input,Alg7Input,AlgResponse,AlgorithmConfig
router = APIRouter(prefix="/valuation", tags=["Valuation Algorithms"])
mock_pi = MockPILayer()
engine = ValuationEngine(mock_pi)
@router.get("/{product_node_id}")
def run_alg1_master(product_node_id: str):
    result = engine.calculate_product_value(product_node_id)
    if result is None:
        raise HTTPException(404, f"Product {product_node_id} not found")
    return {"product_node_id": product_node_id, "estimated_value": result}

@router.post("/alg2/bonding-curve")
def run_alg2(params: BondingCurveParams):
    price = engine._calculate_bonding_curve_price(params)
    return {"algorithm": "ALG-2 Bonding Curve", "price": price}


@router.post("/alg3/multi-factor")
def run_alg3(factors: list[ValuationInputFactor]):
    fake_product = ProductValuation(
        productNodeId="temp",
        algorithm={"algType": "MARKET_COMPARABLE"},
        input_factors=factors
    )
    price = engine._estimate_multi_factor_value(fake_product)
    return {"algorithm": "ALG-3 Multi-Factor", "price": price}

@router.post("/alg4/reputation")
def run_alg4(neutral_price: float, reputation: ReputationSnapshot):
    price = engine._calculate_reputation_weighted_value(neutral_price, reputation)
    return {"algorithm": "ALG-4 Reputation", "price": price}

@router.post("/alg5/dynamic-curve")
def run_alg5(base_price: float, demand: DemandSignal, curve: BondingCurveParams):
    fake_product = ProductValuation(
        productNodeId="temp",
        algorithm={"algType": "BONDING_CURVE", "bondingCurve": curve},
        demandSignal=demand
    )
    final_price = engine._calculate_dynamic_bonding_curve(base_price, fake_product)
    return {"algorithm": "ALG-5 Dynamic Curve", "adjusted_price": final_price}

@router.post("/alg6/yield-factor") 
def run_alg6(input_data: Alg6Input): 
    factor = engine._model_token_yield_as_factor( input_data.staking, input_data.reward, input_data.behavior_score ) 
    return {"algorithm": "ALG-6 Yield Factor", "yield_factor": factor}

@router.post("/alg7/hybrid")
def run_alg7(input_data: Alg7Input):
    engine_factors = []
    if input_data.input_factors:
        for f in input_data.input_factors:
            engine_factors.append(
                ValuationInputFactor(
                    factor_type=f.factor_type,  
                    value=f.score             
                )
            )
    engine_algorithm = ValuationAlgorithm(
        alg_type=input_data.algorithm.type,  
        weights=input_data.algorithm.weights or None,
        bonding_curve=None
    )
    engine_reputation = None
    if input_data.reputation_data:
        engine_reputation = ReputationSnapshot(
            reputationScore=input_data.reputation_data.reputationScore,
            buyerCount=input_data.reputation_data.buyerCount,
            averageRating=input_data.reputation_data.averageRating
        )

    engine_demand = None
    if input_data.demand_signal:
        engine_demand = DemandSignal(
            uniqueBuyersWindow=input_data.demand_signal.demand_index,  # map fields as needed
            volumeWindow=input_data.demand_signal.velocity
        )
    product = ProductValuation(
        productNodeId=input_data.product_node_id,
        algorithm=engine_algorithm,
        input_factors=engine_factors or None,
        reputation_data=engine_reputation,
        demand_signal=engine_demand
    )
    price = engine._calculate_hybrid_value(product)

    return {"algorithm": "ALG-7 Hybrid Composite", "price": price}

