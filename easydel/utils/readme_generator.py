import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from easydel import __version__
from easydel.etils.etils import get_logger

logger = get_logger(__name__)


@dataclass
class ModelInfo:
	"""Model information container."""

	name: str
	type: str
	repo_id: str
	description: Optional[str] = None
	model_class: Optional[str] = None
	features: Optional[list] = None
	supported_tasks: Optional[list] = None
	limitations: Optional[list] = None
	version: str = __version__


class ReadmeGenerator:
	"""Generate README files for EasyDeL models."""

	def __init__(self, template_dir: Optional[str] = None):
		"""
		Initialize the README generator.

		Args:
		    template_dir: Optional custom template directory path
		"""

		# Setup Jinja environment
		if template_dir and os.path.exists(template_dir):
			self.env = Environment(
				loader=FileSystemLoader(template_dir),
				autoescape=select_autoescape(["html", "xml"]),
			)
		else:
			# Use default template
			self.env = Environment(
				loader=FileSystemLoader(os.path.dirname(__file__)),
				autoescape=select_autoescape(["html", "xml"]),
			)

	@property
	def template_path(self):
		return os.path.join(
			os.path.dirname(os.path.abspath(__file__)), "readme_template.jinja"
		)

	def generate_readme(
		self,
		model_info: ModelInfo,
		output_path: Optional[str] = None,
	) -> str:
		"""
		Generate README content for a model.

		Args:
		    model_info: Model information
		    output_path: Optional path to save the README
		    template_name: Name of the template to use

		Returns:
		    Generated README content
		"""
		try:
			template = self.env.get_template("readme_template.jinja")
			content = template.render(model=model_info)

			if output_path:
				output_path = Path(output_path)
				output_path.parent.mkdir(parents=True, exist_ok=True)
				with open(output_path, "w", encoding="utf-8") as f:
					f.write(content)
				logger.info(f"README saved to {output_path}")

			return content

		except Exception as e:
			logger.error(f"Error generating README: {str(e)}")
			raise


# Example usage
if __name__ == "__main__":
	model_info = ModelInfo(
		name="LLaMA-2-7B-EasyDeL",
		type="CausalLM",
		repo_id="erfanzar/LLaMA-2-7B-EasyDeL",
	)

	generator = ReadmeGenerator()
	readme = generator.generate_readme(model_info, "rdm.md")
