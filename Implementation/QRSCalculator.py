class QRSCalculator:

    def __init__(self):
        
        # category weights
        self.weights = {
            "MS": 0.25,   # Medical Symptom Presence
            "SC": 0.25,   # Safety-Critical Situation
            "MD": 0.20,   # Medical Decision Request
            "UE": 0.15,   # Urgency Indicator
            "PA": 0.10,   # Physical Assistance Requirement
            "ES": 0.05    # Emotional State
        }


    def apply_override(self, scores, qrs):

        # Severe symptom → force HIGH boundary
        if scores.get("MS", 0) == 1.0:
            qrs = max(qrs, 0.5)

        # Critical safety event → force HIGH boundary
        if scores.get("SC", 0) == 1.0:
            qrs = max(qrs, 0.5)

        return qrs

    def compute_qrs(self, scores):
        """
        scores: dictionary containing category scores
        example:
        {
            "MS":1,
            "SC":0.5,
            "MD":1,
            "UE":0.5,
            "PA":0,
            "ES":0
        }
        """

        qrs = 0.0

        for category, weight in self.weights.items():

            score = scores.get(category, 0)
            # validation
            assert 0 <= score <= 1, f"Invalid score for {category}: {score}"
            qrs += weight * score

        return round(qrs,3)


    def get_risk_level(self, qrs):

        if qrs < 0.15:
            return "LOW"

        elif qrs < 0.50:
            return "MEDIUM"

        else:
            return "HIGH"


    def evaluate(self, scores):
        """
        Full evaluation pipeline
        """

        qrs = self.compute_qrs(scores)

        # APPLY OVERRIDE HERE
        qrs = self.apply_override(scores, qrs)
        
        level = self.get_risk_level(qrs)

        return {
            "QRS": qrs,
            "Risk_Level": level
        }
    
"""
Example Usage

scores = {
    "MS": 1,
    "SC": 0.5,
    "MD": 1,
    "UE": 0.5,
    "PA": 0,
    "ES": 0
}

qrs_model = QRSCalculator()

result = qrs_model.evaluate(scores)

print(result)

"""