from app.agents.valuation_engine import ValuationEngine, MockPILayer
from app.core.config import settings

class ValuationService:
    def __init__(self):
        # Load PI layer according to env
        if settings.pi_layer_mode == "mock":
            self.pi = MockPILayer()
        else:
            raise NotImplementedError("Only mock PI layer supported currently")

        self.engine = ValuationEngine(pi_layer=self.pi)

    def get_valuation(self, product_id: str):
        return self.engine.calculate_product_value(product_id)
