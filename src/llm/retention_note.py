"""Generate retention notes from churn predictions with an LLM."""

import anthropic
from dotenv import load_dotenv

from src.utils.logger import logger

load_dotenv()

MODEL = "claude-haiku-5-5"

SYSTEM_PROMPT = """You write short notes for a telecom customer retention team.

Rules:
- Use only the facts provided. Do not invent customer details, offers or numbers.
- Describe factors as associated with churn risk, not as causes.
- Write 2-3 sentences: the risk level, the main reasons, and one concrete
  retention action that addresses the most important reason.
- Plain language. Do not mention models or SHAP."""

FEATURE_DESCRIPTIONS = {
    "tenure": "Time as a customer: {value} months",
    "Contract": "Contract type: {value}",
    "InternetService": "Internet service: {value}",
    "PaymentMethod": "Payment method: {value}",
    "MonthlyCharges": "Monthly bill: ${value}",
    "PaperlessBilling": "Paperless billing: {value}",
    "OnlineSecurity": "Online security add-on: {value}",
    "TechSupport": "Tech support add-on: {value}",
    "OnlineBackup": "Online backup add-on: {value}",
    "DeviceProtection": "Device protection add-on: {value}",
    "StreamingTV": "Streaming TV: {value}",
    "StreamingMovies": "Streaming movies: {value}",
    "MultipleLines": "Multiple phone lines: {value}",
    "PhoneService": "Phone service: {value}",
    "SeniorCitizen": "Senior citizen (1 = yes): {value}",
    "Partner": "Has a partner: {value}",
    "Dependents": "Has dependents: {value}",
    "gender": "Gender: {value}",
}


def describe(reason: dict) -> str:
    """Turn one SHAP reason into a plain-English fact."""
    template = FEATURE_DESCRIPTIONS.get(
        reason["feature"], reason["feature"] + ": {value}"
    )
    return f"{template.format(value=reason['value'])} ({reason['effect']})"


def build_prompt(prediction: dict) -> str:
    """Build the user message from a prediction dict."""
    factors = "\n".join(f"- {describe(r)}" for r in prediction["reasons"])
    return (
        f"Churn probability: {prediction['churn_probability']:.0%}\n"
        f"Top factors, most important first:\n{factors}\n\n"
        "Write the retention note."
    )


class RetentionNoteGenerator:
    """Call the LLM to turn a prediction into a retention note."""

    def __init__(self, client=None, model: str = MODEL):
        self.model = model
        try:
            self.client = client or anthropic.Anthropic()
        except Exception as e:
            logger.warning(f"LLM client unavailable, notes disabled: {e}")
            self.client = None

    def generate(self, prediction: dict) -> str | None:
        """Return a retention note, or None if the LLM is unavailable."""
        if self.client is None:
            return None
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_prompt(prediction)}],
            )
            note = "".join(
                block.text for block in response.content if block.type == "text"
            ).strip()
            logger.info(
                f"Retention note generated ({response.usage.input_tokens} in, "
                f"{response.usage.output_tokens} out tokens)"
            )
            return note or None
        except Exception as e:
            logger.warning(f"Retention note generation failed: {e}")
            return None


if __name__ == "__main__":
    from src.pipeline.prediction_pipeline import PredictionPipeline

    customer = {
        "customerID": "DEMO-0001",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
        "TotalCharges": "85.0",
    }
    prediction = PredictionPipeline().predict(customer)
    print(build_prompt(prediction))
    print("\n---\n")
    print(RetentionNoteGenerator().generate(prediction))
