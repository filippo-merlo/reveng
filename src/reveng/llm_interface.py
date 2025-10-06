import logging
from pathlib import Path
from typing import Optional

import litellm
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader, Template
from litellm import completion
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_random_exponential

logger = logging.getLogger(__name__)

ENV_FILE = Path(__file__).parent.parent.parent / ".env"

litellm.enable_json_schema_validation = True


class BaseLLMInterface:
    """Base class for LLM interactions with common functionality."""

    def __init__(
        self,
        model_name: str,
        template_path: Optional[Path | str] = None,
        temperature: float = 1.0,
    ) -> None:
        """
        Args:
            model_name: Identifier understood by ``litellm`` (e.g. ``"gpt-4"``).
            template_path: Optional override for the Jinja prompt template.
            temperature: Forwarded to the model.
        """
        self.model_name = model_name
        self.temperature = temperature

        resolved_template = Path(template_path) if template_path is not None else None
        self._template = (
            self._load_template(resolved_template) if resolved_template else None
        )

        # Load environment variables
        if ENV_FILE.exists():
            load_dotenv(ENV_FILE)
            logger.info(f"Loaded environment variables from {ENV_FILE}")
        else:
            logger.error(
                f"No .env file found at {ENV_FILE}! Please create a .env file in the root of the project."
            )

    @staticmethod
    def _load_template(template_path: Path) -> Template:
        """Load a Jinja2 template from file."""
        env = Environment(
            loader=FileSystemLoader(template_path.parent),
        )
        return env.get_template(template_path.name)

    @retry(
        stop=stop_after_attempt(3),
        # 5-120 seconds between attempts, to help avoid rate limiting
        wait=wait_random_exponential(multiplier=1, min=5, max=120),
        reraise=True,
    )
    def _completion_with_retry(
        self,
        response_format: Optional[BaseModel] = None,
        **kwargs,
    ):
        """Make LLM completion request with retry logic."""
        response = completion(
            **kwargs,
            response_format=response_format,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from model")

        if response_format:
            try:
                return response_format.model_validate_json(content)
            except Exception as exc:
                logger.error(f"Failed to parse response: {exc}")
                logger.debug(
                    f"Raw response content: {response.choices[0].message.content}"
                )
                raise ValueError(f"Invalid response format from model: {exc}")

        return content

    def _make_completion_request(
        self,
        prompt: str,
        response_format: Optional[BaseModel] = None,
        **kwargs,
    ):
        """Make a completion request with standard parameters."""
        try:
            response = self._completion_with_retry(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                response_format=response_format,
                **kwargs,
            )
            return response
        except Exception as exc:
            logger.error(f"Model request failed: {exc}")
            raise

    def render_template(self, **kwargs) -> str:
        """Render the template with given context."""
        if self._template is None:
            raise ValueError("No template provided")
        return self._template.render(**kwargs)
