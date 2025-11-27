## valuation_engine.py
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import math

# --- 1. Data Models (Mirroring our BoB & Solidity structs) ---
# These models ensure that the data flowing into our algorithm is structured and validated.

class AlgorithmType(str, Enum):
    """Defines the types of valuation algorithms the DAO can govern."""
    MARKET_COMPARABLE = "MARKET_COMPARABLE"
    BONDING_CURVE = "BONDING_CURVE"
    REPUTATION_WEIGHTED = "REPUTATION_WEIGHTED"
    HYBRID_COMPOSITE = "HYBRID_COMPOSITE"
    # Add other types as they are developed

class CurveType(str, Enum):
    """Defines the types of bonding curves."""
    LINEAR = "LINEAR"
    POLYNOMIAL = "POLYNOMIAL"
    EXPONENTIAL = "EXPONENTIAL"

class BondingCurveParams(BaseModel):
    """Parameters for a bonding curve, governed by the DAO."""
    curve_type: CurveType = Field(alias="curveType")
    slope: int
    exponent: Optional[int] = None
    base_value: int = Field(alias="baseValue")
    current_supply: int = Field(alias="currentSupply")

class ValuationAlgorithm(BaseModel):
    """The specific algorithm configuration for a product."""
    alg_type: AlgorithmType = Field(alias="algType")
    weights: Optional[List[int]] = None
    bonding_curve: Optional[BondingCurveParams] = Field(alias="bondingCurve", default=None)
    # ... other params like mlOracleAddress can be added here

class ValuationInputFactor(BaseModel):
    """
    Represents a single weighted factor for ALG-3, the multi-factor model.
    """
    factor_type: str
    value: float # The raw or normalized value of the factor

class ReputationSnapshot(BaseModel):
    """
    Represents the reputation data for a seller/product, used by ALG-4.
    """
    reputation_score: float = Field(alias="reputationScore")
    buyer_count: int = Field(alias="buyerCount")
    average_rating: float = Field(alias="averageRating")

class StakingConfig(BaseModel):
    """
    Simplified StakingConfig for valuation input (from RewardsAndStakingAgent).
    """
    minimum_stake_amount: int = Field(alias="minimumStakeAmount")
    base_apy: float = Field(alias="baseAPY") # e.g., 0.05 for 5% APY

class RewardPolicyParams(BaseModel):
    """
    Game-theory related reward parameters (from RewardsAndStakingAgent).
    """
    cooperation_reward_bonus: float = Field(alias="cooperationRewardBonus") # e.g., 0.1 for 10% bonus
    defection_penalty: float = Field(alias="defectionPenalty") # e.g., 0.2 for 20% penalty

class StakingPositionSnapshot(BaseModel):
    """
    Aggregated staking data for a product node, used by ALG-6's valuation aspect.
    """
    total_staked_amount: int = Field(alias="totalStakedAmount")
    # effective_apy: float = Field(alias="effectiveAPY") # Could be derived from StakingConfig
    behavior_score: float = Field(alias="behaviorScore") # e.g., 0.8 for 80% cooperative behavior
    # ... other relevant aggregated staking data


class DemandSignal(BaseModel):
    """
    Represents market demand signals for a product, used by ALG-5.
    """
    unique_buyers_window: int = Field(alias="uniqueBuyersWindow")
    volume_window: float = Field(alias="volumeWindow")

class LiquidityState(BaseModel):
    """
    Represents the liquidity state for a product's token, used by ALG-5.
    """
    liquidity_depth: float = Field(alias="liquidityDepth")
    pool_reserves: float = Field(alias="poolReserves")
    spread_bps: int = Field(alias="spreadBps")

class ProductValuation(BaseModel):
    """
    Represents the complete valuation data for a product, fetched from the PI Layer.
    This is a direct translation of our BTF: ProductValuation.
    """
    product_node_id: str = Field(alias="productNodeId")
    algorithm: ValuationAlgorithm
    # Input factors for multi-factor valuation (ALG-3)
    input_factors: Optional[List[ValuationInputFactor]] = None
    # Reputation data for reputation-weighted valuation (ALG-4)
    reputation_data: Optional[ReputationSnapshot] = Field(alias="reputationData", default=None)
    # Staking and reward policy data for token yield modeling (ALG-6)
    staking_config: Optional[StakingConfig] = Field(alias="stakingConfig", default=None)
    reward_policy_params: Optional[RewardPolicyParams] = Field(alias="rewardPolicyParams", default=None)
    # Market data for dynamic curve valuation (ALG-5)
    demand_signal: Optional[DemandSignal] = Field(alias="demandSignal", default=None)
    liquidity_state: Optional[LiquidityState] = Field(alias="liquidityState", default=None)
    

# --- 2. Mock Persistence & Integration (PI) Layer ---
# This class simulates fetching DAO-governed data from our database.
# Later, we will replace these mock methods with actual SQL queries.

class MockPILayer:
    """
    A mock representation of our Persistence and Integration (PI) layer.
    It returns placeholder data, allowing the algorithm to be developed and tested
    without a live database connection.
    """
    def __init__(self):
        # Sample data store. In reality, this would be a database.
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
                    weights=[80, 20] # 80% from ALG-3/6, 20% from ALG-4
                ),
                # These factors are the input for ALG-3
                input_factors=[
                    ValuationInputFactor(factor_type="infraCost", value=50.5),
                    ValuationInputFactor(factor_type="competitorPriceIndex", value=120.75),
                    ValuationInputFactor(factor_type="regionalMultiplier", value=0.95),
                    ValuationInputFactor(factor_type="daoProfitMargin", value=1.15) # 15% margin
                ],
                # Data needed for ALG-4
                reputationData=ReputationSnapshot(
                    reputationScore=0.95, # e.g., a score of 95 out of 100
                    buyerCount=500,
                    averageRating=4.8
                ),
                # Data needed for ALG-6's valuation aspect
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
                    algType=AlgorithmType.HYBRID_COMPOSITE, # Using hybrid to compose ALG-2 and ALG-5
                    bondingCurve=BondingCurveParams(
                        curveType=CurveType.LINEAR,
                        slope=20,
                        baseValue=500,
                        currentSupply=1000
                    )
                ),
                # High demand signals for this product
                demandSignal=DemandSignal(
                    uniqueBuyersWindow=150, # 150 buyers in the last window
                    volumeWindow=30000.0
                ),
                # Shallow liquidity for this product's token
                liquidityState=LiquidityState(
                    liquidityDepth=5000.0, # Low liquidity
                    poolReserves=10000.0,
                    spreadBps=200 # 2% spread
                )
            ),
        }

    def get_product_valuation(self, product_node_id: str) -> Optional[ProductValuation]:
        """
        MOCK: Fetches the governed valuation model for a product.
        PI SQL equivalent: SELECT algorithmJson, ... FROM ProductValuation WHERE productNodeId = :nodeId;
        """
        print(f"MOCK PI: Fetching valuation for {product_node_id}...")
        return self._product_valuations.get(product_node_id)


# --- 3. The Valuation Engine and ALG-1 Implementation ---

class ValuationEngine:
    """
    The off-chain agent responsible for running valuation and pricing algorithms.
    """
    def __init__(self, pi_layer: MockPILayer):
        self.pi_layer = pi_layer

    def _calculate_bonding_curve_price(self, params: BondingCurveParams) -> float:
        """
        This is the implementation of ALG-2.

        Calculates a price based on a bonding curve's parameters. It defines how
        price moves with circulating supply for fully tokenized products.
        """
        print("--> Executing ALG-2 (Bonding Curve) logic...")
        
        # For calculations, it's often easier to work with normalized supply or simple integers.
        # A real-world implementation would use fixed-point math to avoid floating point errors,
        # especially when mirroring Solidity logic.
        supply = params.current_supply

        if params.curve_type == CurveType.LINEAR:
            # Formula: price = baseValue + (slope * supply)
            price = params.base_value + (params.slope * supply) / 1000 # Use 1000 for better precision with integer slopes
            return round(price, 4)
        
        if params.curve_type == CurveType.POLYNOMIAL:
            # Formula: price = baseValue + slope * (supply ^ exponent)
            if params.exponent is None:
                raise ValueError("Exponent is required for a polynomial curve.")
            price = params.base_value + params.slope * (supply ** params.exponent)
            return round(price, 4)

        if params.curve_type == CurveType.EXPONENTIAL:
            # Formula: price = baseValue * (1 + slope/100) ^ supply
            # This models compounding growth.
            growth_rate = 1 + (params.slope / 100)
            price = params.base_value * math.pow(growth_rate, supply)
            return round(price, 4)

        raise NotImplementedError(f"Bonding curve type {params.curve_type} not implemented.")

    def _estimate_multi_factor_value(self, factors: List[ValuationInputFactor]) -> float:
        """
        This is the implementation of ALG-3.

        Calculates a value by combining multiple input factors. This is a simplified
        example. A real implementation might involve normalization and more complex weighting.
        """
        print("--> Executing ALG-3 (Multi-Factor Estimation) logic...")
        
        # For this example, we'll find the core cost and apply multipliers.
        # This logic can be made as complex as needed.
        factor_map = {f.factor_type: f.value for f in factors}

        # Start with a base cost (e.g., from infrastructure or competitor data)
        base_cost = (factor_map.get("infraCost", 0.0) + factor_map.get("competitorPriceIndex", 0.0)) / 2
        if base_cost == 0:
             print("WARN: Base cost for multi-factor valuation is zero.")
             return 0.0

        # Apply multipliers
        regional_multiplier = factor_map.get("regionalMultiplier", 1.0)
        profit_margin = factor_map.get("daoProfitMargin", 1.0)

        final_value = base_cost * regional_multiplier * profit_margin
        return final_value

    def _model_token_yield_as_factor(self, staking_config: StakingConfig, reward_policy: RewardPolicyParams, behavior_score: float) -> float:
        """
        This function models the 'token yield' aspect of ALG-6 for valuation.
        It calculates an incentive factor based on staking APY and game-theory adjustments.
        """
        print("--> Modeling ALG-6 (Token Yield Factor) logic...")
        
        # Start with the base APY as a primary driver of yield
        yield_factor = staking_config.base_apy

        # Adjust based on cooperative behavior
        # If behavior_score is high (e.g., 1.0), apply bonus. If low (e.g., 0.0), apply penalty.
        if behavior_score >= 0.7: # Assume 0.7 is a threshold for 'cooperative'
            yield_factor += reward_policy.cooperation_reward_bonus
        elif behavior_score < 0.3: # Assume 0.3 is a threshold for 'defection'
            yield_factor -= reward_policy.defection_penalty
        
        # Ensure yield factor doesn't go negative or too high
        yield_factor = max(0.01, min(yield_factor, 0.5)) # Cap between 1% and 50% for realism

        print(f"    Modeled Token Yield Factor: {yield_factor:.2f}")
        return yield_factor


    def _estimate_multi_factor_value(self, valuation_data: ProductValuation) -> float:
        """
        This is the implementation of ALG-3.

        Calculates a value by combining multiple input factors, now including token yield.
        """
        print("--> Executing ALG-3 (Multi-Factor Estimation) logic...")
        
        factor_map = {f.factor_type: f.value for f in valuation_data.input_factors or []}

        # Incorporate Token Yield Factor from ALG-6 modeling
        token_yield_factor = 0.0 # Default to no yield
        if valuation_data.staking_config and valuation_data.reward_policy_params:
            # For simplicity, we'll assume a fixed behavior score for the product node itself
            # In a real system, this would come from aggregated behavior data.
            product_behavior_score = 0.85 # Example: 85% cooperative behavior for this product's stakers
            token_yield_factor = self._model_token_yield_as_factor(
                valuation_data.staking_config,
                valuation_data.reward_policy_params,
                product_behavior_score
            )
            # Add this as a factor to the map, or directly use it.
            # For now, let's use it as a positive multiplier to the profit margin.
            factor_map["tokenYieldFactor"] = 1 + token_yield_factor # e.g., 1.10 for 10% yield

        # Start with a base cost (e.g., from infrastructure or competitor data)
        base_cost = (factor_map.get("infraCost", 0.0) + factor_map.get("competitorPriceIndex", 0.0)) / 2
        if base_cost == 0:
             print("WARN: Base cost for multi-factor valuation is zero.")
             return 0.0

        # Apply multipliers
        regional_multiplier = factor_map.get("regionalMultiplier", 1.0)
        profit_margin = factor_map.get("daoProfitMargin", 1.0) * factor_map.get("tokenYieldFactor", 1.0) # Apply yield here

        final_value = base_cost * regional_multiplier * profit_margin
        return final_value

    def _calculate_reputation_weighted_value(self, neutral_price: float, reputation: ReputationSnapshot) -> float:
        """
        This is the implementation of ALG-4.

        It now calculates an independent value based on reputation, using a
        neutral price (like the multi-factor value) as a baseline.
        """
        print("--> Executing ALG-4 (Reputation Weighted Value) logic...")

        # Create a modifier based on reputation score, volume, and rating.
        # A modifier of 1.0 is neutral.
        
        # A reputation score of 0.95 implies a 5% discount from a neutral 1.0
        # A score of 1.05 implies a 5% premium.
        # Let's model the modifier as being centered around 1.0.
        # modifier = 1.0 + (reputation.reputation_score - 1.0)
        # For a score of 0.95, this would be 1.0 + (0.95 - 1.0) = 0.95.
        # Let's use the simpler direct score.
        modifier = reputation.reputation_score

        # Add a small bonus for a high number of buyers (demand signal)
        if reputation.buyer_count > 400:
            modifier += 0.05 # +5% bonus

        # Add a small bonus for excellent ratings
        if reputation.average_rating > 4.7:
            modifier += 0.05 # +5% bonus

        print(f"    Reputation modifier calculated: {modifier:.2f}")
        return neutral_price * modifier

    def _calculate_dynamic_bonding_curve(self, base_price: float, valuation_data: ProductValuation) -> float:
        """
        This is the implementation of ALG-5.

        Adjusts a bonding curve price based on market-sensitive factors like
        demand-supply ratio (DSR) and liquidity depth.
        """
        print("--> Executing ALG-5 (Dynamic Bonding Curve) logic...")
        if not valuation_data.demand_signal or not valuation_data.algorithm.bonding_curve:
            print("    WARN: Missing demand signal or bonding curve params for ALG-5. Skipping.")
            return base_price

        demand = valuation_data.demand_signal.unique_buyers_window
        supply = valuation_data.algorithm.bonding_curve.current_supply

        # Calculate Demand-to-Supply Ratio (DSR)
        # To avoid wild swings, we can use a portion of the supply for the ratio.
        # Let's assume the "active" supply is 10% of total supply.
        active_supply = supply * 0.1
        if active_supply == 0:
            return base_price
            
        dsr = demand / active_supply
        print(f"    Demand-to-Supply Ratio (DSR): {dsr:.2f}")

        # Adjust price based on DSR. A DSR > 1 means demand is high, pushing price up.
        # We use a logarithm to dampen the effect and prevent extreme price swings.
        # A DSR of 1 results in a modifier of 1 (no change).
        dsr_modifier = 1 + (math.log(dsr) if dsr > 0 else 0) * 0.1 # 10% of the log of DSR

        print(f"    DSR Modifier: {dsr_modifier:.2f}")
        return base_price * dsr_modifier

    def _calculate_hybrid_value(self, valuation_data: ProductValuation) -> float:
        """
        This is the implementation of ALG-7. Calculates a hybrid composite value.
        It composes multiple algorithm outputs using DAO-governed weights.
        """
        print("--> Executing ALG-7 (Hybrid Composite) logic...")

        # --- Playbook for a Factor/Reputation Hybrid (product-002-hybrid) ---
        if valuation_data.input_factors and valuation_data.reputation_data:
            weights = valuation_data.algorithm.weights
            if not weights or len(weights) < 2:
                raise ValueError("Hybrid algorithm requires at least two weights.")

            # 1. Calculate Component 1: Multi-factor value (ALG-3 + ALG-6 aspect)
            factor_value = self._estimate_multi_factor_value(valuation_data)
            print(f"    ALG-3/6 Component Value: {factor_value:.2f}")

            # 2. Calculate Component 2: Reputation value (ALG-4)
            # ALG-4 uses the factor_value as a neutral baseline to calculate its adjustment.
            reputation_value = self._calculate_reputation_weighted_value(factor_value, valuation_data.reputation_data)
            print(f"    ALG-4 Component Value: {reputation_value:.2f}")

            # 3. Combine components using DAO-governed weights
            # Formula: estimatedValue = (factor_value * w1 + reputation_value * w2) / (w1 + w2)
            total_weight = sum(weights)
            if total_weight == 0:
                raise ValueError("Total weight for hybrid algorithm cannot be zero.")

            weighted_value = (factor_value * weights[0] + reputation_value * weights[1]) / total_weight
            print(f"    Combined weighted value (80/20 split): {weighted_value:.2f}")
            base_value = weighted_value

        # --- Playbook for a Dynamic Bonding Curve Hybrid (product-005-dynamic) ---
        elif valuation_data.algorithm.bonding_curve and valuation_data.demand_signal:
            static_curve_price = self._calculate_bonding_curve_price(valuation_data.algorithm.bonding_curve)
            base_value = self._calculate_dynamic_bonding_curve(static_curve_price, valuation_data)
        else:
            raise NotImplementedError("No valid playbook found for this hybrid product's data.")

        return round(base_value, 2)

    def calculate_product_value(self, product_node_id: str) -> Optional[float]:
        """
        This is the implementation of ALG-1 (Marketplace Core).

        It fetches the DAO-governed valuation model for a product and chooses the
        correct sub-algorithm to calculate the estimated value.
        """
        print(f"\n--- Running ALG-1: calculateProductValue for {product_node_id} ---")
        
        # 1. Fetch the governed valuation model from the PI layer.
        valuation_data = self.pi_layer.get_product_valuation(product_node_id)

        if not valuation_data:
            print(f"ERROR: No valuation data found for {product_node_id}")
            return None

        alg_type = valuation_data.algorithm.alg_type
        print(f"DAO-governed algorithm type is: {alg_type}")

        # 2. Choose and execute the appropriate sub-algorithm based on algType.
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