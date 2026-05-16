"""
Diffing module for analyzing differences between base and finetuned models.

Keep the package root lightweight so subpackages such as
`diffing.logit_lens_methods` can be imported without eagerly importing the
older donor toolkit modules and their optional dependencies.
"""

__all__ = ["methods", "evaluators", "logit_lens_methods"]
