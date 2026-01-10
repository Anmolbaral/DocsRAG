"""
Canonicalization Layer for Claims.

Normalizes claim values (especially skills) to ensure consistent naming
across documents. Prevents duplicates like "React.js" vs "React" vs "ReactJS".
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# Default skill normalization map
DEFAULT_SKILL_ALIASES = {
    # JavaScript ecosystem
    "react.js": "React",
    "reactjs": "React",
    "react js": "React",
    "next.js": "Next.js",
    "nextjs": "Next.js",
    "node.js": "Node.js",
    "nodejs": "Node.js",
    "vue.js": "Vue",
    "vuejs": "Vue",
    # Python ecosystem
    "python3": "Python",
    "python 3": "Python",
    "python2": "Python",
    "scikit-learn": "Scikit-Learn",
    "sklearn": "Scikit-Learn",
    "sci-kit learn": "Scikit-Learn",
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow",
    # DevOps
    "ci/cd": "CI_CD",
    "ci cd": "CI_CD",
    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "k8s": "Kubernetes",
    # Databases
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "mongo": "MongoDB",
    "mongodb": "MongoDB",
    "mysql": "MySQL",
    # Cloud
    "aws": "AWS",
    "amazon web services": "AWS",
    "gcp": "GCP",
    "google cloud": "GCP",
    "google cloud platform": "GCP",
    "azure": "Azure",
    "microsoft azure": "Azure",
    # Languages
    "javascript": "JavaScript",
    "js": "JavaScript",
    "typescript": "TypeScript",
    "ts": "TypeScript",
    "c++": "C++",
    "cpp": "C++",
    "c#": "C#",
    "csharp": "C#",
    # ML/AI
    "machine learning": "Machine Learning",
    "ml": "Machine Learning",
    "artificial intelligence": "AI",
    "deep learning": "Deep Learning",
    "natural language processing": "NLP",
    "computer vision": "Computer Vision",
    "cv": "Computer Vision",
    # Web
    "html5": "HTML5",
    "css3": "CSS3",
    "scss": "SCSS",
    "sass": "SASS",
    "rest api": "REST API",
    "restful": "REST API",
    "graphql": "GraphQL",
    # Tools
    "git": "Git",
    "github": "GitHub",
    "gitlab": "GitLab",
    "jira": "Jira",
    "vscode": "VSCode",
    "visual studio code": "VSCode",
}


class Canonicalizer:
    """
    Normalizes claim values to canonical forms.

    Handles:
    - Skill name normalization
    - Case normalization
    - Whitespace trimming
    - Custom alias mapping
    """

    def __init__(self, custom_map_path: Optional[str] = None):
        """
        Initialize canonicalizer.

        Args:
            custom_map_path: Optional path to custom canon_map.json
        """
        self.skill_map = DEFAULT_SKILL_ALIASES.copy()

        # Load custom mappings if provided
        if custom_map_path and Path(custom_map_path).exists():
            self._load_custom_map(custom_map_path)

    def _load_custom_map(self, path: str) -> None:
        """Load custom canonicalization map."""
        try:
            with open(path, "r") as f:
                custom_map = json.load(f)

            # Merge with defaults (custom overrides default)
            self.skill_map.update(custom_map.get("skill_aliases", {}))
            logger.info(f"Loaded {len(custom_map)} custom mappings from {path}")

        except Exception as e:
            logger.warning(f"Could not load custom map from {path}: {e}")

    def save_custom_map(self, path: str) -> None:
        """Save current mappings to file."""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            data = {"skill_aliases": self.skill_map, "version": "1.0"}

            with open(path, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved canonicalization map to {path}")

        except Exception as e:
            logger.error(f"Error saving canonicalization map: {e}")

    def canonicalize_skill(self, skill: str) -> str:
        """
        Normalize a skill name to canonical form.

        Args:
            skill: Raw skill name

        Returns:
            Canonical skill name
        """
        # Trim whitespace
        skill = skill.strip()

        # Check lowercase version in map
        lowercase = skill.lower()
        if lowercase in self.skill_map:
            return self.skill_map[lowercase]

        # If not in map, return title case as default
        return skill.title() if skill.islower() else skill

    def canonicalize_value(self, claim_type: str, value: str) -> str:
        """
        Normalize a claim value based on its type.

        Args:
            claim_type: Type of claim (skill, value, etc.)
            value: Raw value

        Returns:
            Canonical value
        """
        if claim_type == "skill":
            return self.canonicalize_skill(value)

        # For other types, just trim and normalize case
        value = value.strip()

        if claim_type in ["value", "soft_skill"]:
            # Title case for values/soft skills
            return value.title() if value.islower() else value

        # Default: return as-is
        return value

    def add_alias(self, alias: str, canonical: str) -> None:
        """
        Add a new alias mapping.

        Args:
            alias: Alias to map from (will be lowercased)
            canonical: Canonical form to map to
        """
        self.skill_map[alias.lower()] = canonical
        logger.debug(f"Added alias: {alias} -> {canonical}")

    def get_all_canonical_skills(self) -> set:
        """Get set of all canonical skill names."""
        return set(self.skill_map.values())


# Singleton instance for convenience
_default_canonicalizer: Optional[Canonicalizer] = None


def get_canonicalizer() -> Canonicalizer:
    """Get the default canonicalizer instance."""
    global _default_canonicalizer
    if _default_canonicalizer is None:
        _default_canonicalizer = Canonicalizer()
    return _default_canonicalizer
