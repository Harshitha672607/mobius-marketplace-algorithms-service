## valuation_engine.py
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import math

class AlgorithmType(str, Enum):
    MARKET_COMPARABLE = "MARKET_COMPARABLE"
    BONDING_CURVE = "BONDING_CURVE"
    REPUTATION_WEIGHTED = "REPUTATION_WEIGHTED"
    HYBRID_COMPOSITE = "HYBRID_COMPOSITE"

class CurveType(str, Enum):
    LINEAR = "LINEAR"
    POLYNOMIAL = "POLYNOMIAL"
    EXPONENTIAL = "EXPONENTIAL"

class BondingCurveParams(BaseModel):
    curve_type: CurveType = Field(alias="curveType")
    slope: int
    exponent: Optional[int] = None
    base_value: int = Field(alias="baseValue")
    current_supply: int = Field(alias="currentSupply")

class ValuationAlgorithm(BaseModel):
    alg_type: AlgorithmType = Field(alias="algType")
    weights: Optional[List[int]] = None
    bonding_curve: Optional[BondingCurveParams] = Field(alias="bondingCurve", default=None)


class ValuationInputFactor(BaseModel):
    factor_type: str
    value: float 

class ReputationSnapshot(BaseModel):
    reputation_score: float = Field(alias="reputationScore")
    buyer_count: int = Field(alias="buyerCount")
    average_rating: float = Field(alias="averageRating")

class StakingConfig(BaseModel):
    minimum_stake_amount: int = Field(alias="minimumStakeAmount")
    base_apy: float = Field(alias="baseAPY")

class RewardPolicyParams(BaseModel):
    cooperation_reward_bonus: float = Field(alias="cooperationRewardBonus") 
    defection_penalty: float = Field(alias="defectionPenalty") 

class StakingPositionSnapshot(BaseModel):
    total_staked_amount: int = Field(alias="totalStakedAmount")
    behavior_score: float = Field(alias="behaviorScore") 


class DemandSignal(BaseModel):
    unique_buyers_window: int = Field(alias="uniqueBuyersWindow")
    volume_window: float = Field(alias="volumeWindow")

class LiquidityState(BaseModel):
    liquidity_depth: float = Field(alias="liquidityDepth")
    pool_reserves: float = Field(alias="poolReserves")
    spread_bps: int = Field(alias="spreadBps")

class ProductValuation(BaseModel):
    product_node_id: str = Field(alias="productNodeId")
    algorithm: ValuationAlgorithm
    input_factors: Optional[List[ValuationInputFactor]] = None
    reputation_data: Optional[ReputationSnapshot] = Field(alias="reputationData", default=None)
    staking_config: Optional[StakingConfig] = Field(alias="stakingConfig", default=None)
    reward_policy_params: Optional[RewardPolicyParams] = Field(alias="rewardPolicyParams", default=None)
    demand_signal: Optional[DemandSignal] = Field(alias="demandSignal", default=None)
    liquidity_state: Optional[LiquidityState] = Field(alias="liquidityState", default=None)
    



class MockPILayer:
    def __init__(self):
        self._product_valuations: Dict[str, ProductValuation] = {
            "product-001-bonding": ProductValuation(
                productNodeId="product-001-bonding",
                algorithm=ValuationAlgorithm(
                    algType=AlgorithmType.BONDING_CURVE,
                    bondingCurve=BondingCurveParams(
                        curveType=CurveType.LINEAR,
                        slope=10,
                        baseValue=100,
                        currentSupply=5000
                    )
                )
            ),
            "product-002-hybrid": ProductValuation(
                productNodeId="product-002-hybrid",
                algorithm=ValuationAlgorithm(
                    algType=AlgorithmType.HYBRID_COMPOSITE,
                    weights=[80, 20]
                ),

                input_factors=[
                    ValuationInputFactor(factor_type="infraCost", value=50.5),
                    ValuationInputFactor(factor_type="competitorPriceIndex", value=120.75),
                    ValuationInputFactor(factor_type="regionalMultiplier", value=0.95),
                    ValuationInputFactor(factor_type="daoProfitMargin", value=1.15) # 15% margin
                ],
        
                reputationData=ReputationSnapshot(
                    reputationScore=0.95, 
                    buyerCount=500,
                    averageRating=4.8
                ),
                stakingConfig=StakingConfig(
                    minimumStakeAmount=1000,
                    baseAPY=0.08 # 8% base APY
                ),
                rewardPolicyParams=RewardPolicyParams(
                    cooperationRewardBonus=0.10, # 10% bonus for cooperation
                    defectionPenalty=0.20 # 20% penalty for defection
                )
            ),
            "product-003-poly": ProductValuation(
                productNodeId="product-003-poly",
                algorithm=ValuationAlgorithm(
                    algType=AlgorithmType.BONDING_CURVE,
                    bondingCurve=BondingCurveParams(
                        curveType=CurveType.POLYNOMIAL,
                        slope=2, # Represents coefficient 'a' in ax^n
                        baseValue=50,
                        currentSupply=100,
                        exponent=2 # The 'n' in ax^n
                    )
                )
            ),
            "product-004-exp": ProductValuation(
                productNodeId="product-004-exp",
                algorithm=ValuationAlgorithm(
                    algType=AlgorithmType.BONDING_CURVE,
                    bondingCurve=BondingCurveParams(
                        curveType=CurveType.EXPONENTIAL,
                        slope=5, # Represents growth rate
                        baseValue=100,
                        currentSupply=10
                    )
                )
            ),
            "product-005-dynamic": ProductValuation(
                productNodeId="product-005-dynamic",
                algorithm=ValuationAlgorithm(
                    algType=AlgorithmType.HYBRID_COMPOSITE, 
                    bondingCurve=BondingCurveParams(
                        curveType=CurveType.LINEAR,
                        slope=20,
                        baseValue=500,
                        currentSupply=1000
                    )
                ),
                demandSignal=DemandSignal(
                    uniqueBuyersWindow=150, 
                    volumeWindow=30000.0
                ),
                liquidityState=LiquidityState(
                    liquidityDepth=5000.0, 
                    poolReserves=10000.0,
                    spreadBps=200 
                )
            ),
        }

    def get_product_valuation(self, product_node_id: str) -> Optional[ProductValuation]:
        print(f"MOCK PI: Fetching valuation for {product_node_id}...")
        return self._product_valuations.get(product_node_id)


class ValuationEngine:
    def __init__(self, pi_layer: MockPILayer):
        self.pi_layer = pi_layer

    def _calculate_bonding_curve_price(self, params: BondingCurveParams) -> float:
        print("--> Executing ALG-2 (Bonding Curve) logic...")
        supply = params.current_supply

        if params.curve_type == CurveType.LINEAR:
            price = params.base_value + (params.slope * supply) / 1000 # Use 1000 for better precision with integer slopes
            return round(price, 4)
        
        if params.curve_type == CurveType.POLYNOMIAL:
            if params.exponent is None:
                raise ValueError("Exponent is required for a polynomial curve.")
            price = params.base_value + params.slope * (supply ** params.exponent)
            return round(price, 4)

        if params.curve_type == CurveType.EXPONENTIAL:
            growth_rate = 1 + (params.slope / 100)
            price = params.base_value * math.pow(growth_rate, supply)
            return round(price, 4)

        raise NotImplementedError(f"Bonding curve type {params.curve_type} not implemented.")

    def _estimate_multi_factor_value(self, factors: List[ValuationInputFactor]) -> float:
        print("--> Executing ALG-3 (Multi-Factor Estimation) logic...")
        factor_map = {f.factor_type: f.value for f in factors}
        base_cost = (factor_map.get("infraCost", 0.0) + factor_map.get("competitorPriceIndex", 0.0)) / 2
        if base_cost == 0:
             print("WARN: Base cost for multi-factor valuation is zero.")
             return 0.0
        regional_multiplier = factor_map.get("regionalMultiplier", 1.0)
        profit_margin = factor_map.get("daoProfitMargin", 1.0)

        final_value = base_cost * regional_multiplier * profit_margin
        return final_value

    def _model_token_yield_as_factor(self, staking_config: StakingConfig, reward_policy: RewardPolicyParams, behavior_score: float) -> float:
        print("--> Modeling ALG-6 (Token Yield Factor) logic...")
        yield_factor = staking_config.base_apy
        if behavior_score >= 0.7:
            yield_factor += reward_policy.cooperation_reward_bonus
        elif behavior_score < 0.3:
            yield_factor -= reward_policy.defection_penalty
        yield_factor = max(0.01, min(yield_factor, 0.5)) 

        print(f"    Modeled Token Yield Factor: {yield_factor:.2f}")
        return yield_factor


    def _estimate_multi_factor_value(self, valuation_data: ProductValuation) -> float:
        print("--> Executing ALG-3 (Multi-Factor Estimation) logic...")
        
        factor_map = {f.factor_type: f.value for f in valuation_data.input_factors or []}
        token_yield_factor = 0.0 # Default to no yield
        if valuation_data.staking_config and valuation_data.reward_policy_params:
            product_behavior_score = 0.85
            token_yield_factor = self._model_token_yield_as_factor(
                valuation_data.staking_config,
                valuation_data.reward_policy_params,
                product_behavior_score
            )
            factor_map["tokenYieldFactor"] = 1 + token_yield_factor # e.g., 1.10 for 10% yield

        base_cost = (factor_map.get("infraCost", 0.0) + factor_map.get("competitorPriceIndex", 0.0)) / 2
        if base_cost == 0:
             print("WARN: Base cost for multi-factor valuation is zero.")
             return 0.0

        regional_multiplier = factor_map.get("regionalMultiplier", 1.0)
        profit_margin = factor_map.get("daoProfitMargin", 1.0) * factor_map.get("tokenYieldFactor", 1.0) # Apply yield here

        final_value = base_cost * regional_multiplier * profit_margin
        return final_value

    def _calculate_reputation_weighted_value(self, neutral_price: float, reputation: ReputationSnapshot) -> float:
        print("--> Executing ALG-4 (Reputation Weighted Value) logic...")
        modifier = reputation.reputation_score
        if reputation.buyer_count > 400:
            modifier += 0.05 # +5% bonus
        if reputation.average_rating > 4.7:
            modifier += 0.05 # +5% bonus

        print(f"    Reputation modifier calculated: {modifier:.2f}")
        return neutral_price * modifier

    def _calculate_dynamic_bonding_curve(self, base_price: float, valuation_data: ProductValuation) -> float:
        print("--> Executing ALG-5 (Dynamic Bonding Curve) logic...")
        if not valuation_data.demand_signal or not valuation_data.algorithm.bonding_curve:
            print("    WARN: Missing demand signal or bonding curve params for ALG-5. Skipping.")
            return base_price

        demand = valuation_data.demand_signal.unique_buyers_window
        supply = valuation_data.algorithm.bonding_curve.current_supply
        active_supply = supply * 0.1
        if active_supply == 0:
            return base_price
            
        dsr = demand / active_supply
        print(f"    Demand-to-Supply Ratio (DSR): {dsr:.2f}")
        dsr_modifier = 1 + (math.log(dsr) if dsr > 0 else 0) * 0.1 

        print(f"    DSR Modifier: {dsr_modifier:.2f}")
        return base_price * dsr_modifier

    # def _calculate_hybrid_value(self, valuation_data: ProductValuation) -> float:
    #     print("--> Executing ALG-7 (Hybrid Composite) logic...")
    #     if valuation_data.input_factors and valuation_data.reputation_data:
    #         weights = valuation_data.algorithm.weights
    #         if not weights or len(weights) < 2:
    #             raise ValueError("Hybrid algorithm requires at least two weights.")
    #         factor_value = self._estimate_multi_factor_value(valuation_data)
    #         print(f"    ALG-3/6 Component Value: {factor_value:.2f}")
    #         reputation_value = self._calculate_reputation_weighted_value(factor_value, valuation_data.reputation_data)
    #         print(f"    ALG-4 Component Value: {reputation_value:.2f}")
    #         total_weight = sum(weights)
    #         if total_weight == 0:
    #             raise ValueError("Total weight for hybrid algorithm cannot be zero.")

    #         weighted_value = (factor_value * weights[0] + reputation_value * weights[1]) / total_weight
    #         print(f"    Combined weighted value (80/20 split): {weighted_value:.2f}")
    #         base_value = weighted_value

    #     elif valuation_data.algorithm.bonding_curve and valuation_data.demand_signal:
    #         static_curve_price = self._calculate_bonding_curve_price(valuation_data.algorithm.bonding_curve)
    #         base_value = self._calculate_dynamic_bonding_curve(static_curve_price, valuation_data)
    #     else:
    #         raise NotImplementedError("No valid playbook found for this hybrid product's data.")

    #     return round(base_value, 2)
    def _calculate_hybrid_value(self, valuation_data: ProductValuation) -> float:
        print("--> Executing ALG-7 (Hybrid Composite) logic...")
        if valuation_data.input_factors and valuation_data.reputation_data:
            # ALG-7 Playbook 1: Factor/Reputation Hybrid
            weights = valuation_data.algorithm.weights
            if not weights or len(weights) < 2:
                raise ValueError("Hybrid algorithm requires at least two weights.")
            
            factor_value = self._estimate_multi_factor_value(valuation_data)
            print(f"    ALG-3/6 Component Value: {factor_value:.2f}")
            
            reputation_value = self._calculate_reputation_weighted_value(factor_value, valuation_data.reputation_data)
            print(f"    ALG-4 Component Value: {reputation_value:.2f}")
            
            total_weight = sum(weights)
            if total_weight == 0:
                raise ValueError("Total weight for hybrid algorithm cannot be zero.")
            
            weighted_value = (factor_value * weights[0] + reputation_value * weights[1]) / total_weight
            print(f"    Combined weighted value (80/20 split): {weighted_value:.2f}")
            base_value = weighted_value

        elif valuation_data.algorithm.bonding_curve and valuation_data.demand_signal:
            # ALG-7 Playbook 2: Dynamic Bonding Curve Hybrid
            static_curve_price = self._calculate_bonding_curve_price(valuation_data.algorithm.bonding_curve)
            base_value = self._calculate_dynamic_bonding_curve(static_curve_price, valuation_data)
        else:
            raise NotImplementedError("No valid playbook found for this hybrid product's data.")

        return round(base_value, 2)

    def calculate_product_value(self, product_node_id: str) -> Optional[float]:
        print(f"\n--- Running ALG-1: calculateProductValue for {product_node_id} ---")
        valuation_data = self.pi_layer.get_product_valuation(product_node_id)

        if not valuation_data:
            print(f"ERROR: No valuation data found for {product_node_id}")
            return None

        alg_type = valuation_data.algorithm.alg_type
        print(f"DAO-governed algorithm type is: {alg_type}")

        estimated_value = 0.0
        if alg_type == AlgorithmType.BONDING_CURVE:
            if not valuation_data.algorithm.bonding_curve:
                print("ERROR: Bonding curve parameters are missing.")
                return None
            estimated_value = self._calculate_bonding_curve_price(valuation_data.algorithm.bonding_curve)
        elif alg_type == AlgorithmType.HYBRID_COMPOSITE:
            estimated_value = self._calculate_hybrid_value(valuation_data)
        else:
            raise NotImplementedError(f"Algorithm type {alg_type} is not implemented in this engine.")

        print(f"SUCCESS: Estimated value for {product_node_id} is: {estimated_value}")
        print("--- ALG-1 Finished ---")
        return estimated_value