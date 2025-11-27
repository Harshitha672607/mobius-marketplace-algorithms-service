from pydantic import BaseModel, Field
from typing import Optional, List
from app.agents.valuation_engine import StakingConfig, RewardPolicyParams

# ------------------ COMMON RESPONSE ------------------
class AlgResponse(BaseModel):
    product_node_id: str
    estimated_value: Optional[float]
    message: Optional[str] = "Success"


# ------------------ ALG-6 INPUT ------------------
class Alg6Input(BaseModel):
    staking: StakingConfig
    reward: RewardPolicyParams
    behavior_score: float


# ------------------ FACTOR INPUT ------------------
class FactorInput(BaseModel):
    factor_type: str = Field(..., description="Type of factor, e.g., cost, profit, multiplier")
    score: float = Field(..., description="Score or value for this factor")
    weight: float = Field(..., description="Weight assigned to the factor")


# ------------------ REPUTATION DATA ------------------
class ReputationData(BaseModel):
    reputationScore: float = Field(..., description="Reputation score, e.g., 1.0 = neutral")
    buyerCount: int = Field(..., description="Number of buyers")
    averageRating: float = Field(..., description="Average rating of product/seller")


# ------------------ BONDING CURVE DATA ------------------
class BondingCurveData(BaseModel):
    k: float = 1.0
    m: float = 1.0


# ------------------ DEMAND SIGNAL DATA ------------------
class DemandSignalData(BaseModel):
    demand_index: float
    velocity: float


# ------------------ ALGORITHM CONFIG ------------------
class AlgorithmConfig(BaseModel):
    type: str
    weights: Optional[List[float]] = None
    bonding_curve: Optional[BondingCurveData] = None


# ------------------ ALG-7 INPUT ------------------
class Alg7Input(BaseModel):
    product_node_id: str
    algorithm: AlgorithmConfig

    input_factors: Optional[List[FactorInput]] = None
    reputation_data: Optional[ReputationData] = None
    demand_signal: Optional[DemandSignalData] = None
