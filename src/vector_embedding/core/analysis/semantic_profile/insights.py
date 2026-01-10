"""
Deterministic Insight Engine.

Generates insights from atomic claims using pure logic (no LLM interpretation).
All insights are traceable to source claims.
"""

import logging
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict

from ..claims_db import ClaimsDatabase

logger = logging.getLogger(__name__)


class InsightEngine:
    """
    Generates deterministic insights from claims database.

    All methods use pure logic and statistics - no LLM interpretation.
    Results are reproducible and auditable.
    """

    def __init__(self, db_path: str):
        """
        Initialize insight engine.

        Args:
            db_path: Path to claims.json database
        """
        self.db = ClaimsDatabase(db_path)
        logger.info(f"Loaded {len(self.db)} claims from database")

    def get_top_skills(
        self,
        min_confidence: float = 0.7,
        limit: int = 20,
        context_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get top skills ranked by frequency and confidence.

        Args:
            min_confidence: Minimum confidence threshold
            limit: Max number of skills to return
            context_filter: Optional context filter (production, academic, etc.)

        Returns:
            List of skill dicts with value, count, avg_confidence, contexts
        """
        # Query skill claims
        filters = {"claim_type": "skill", "confidence": {"$gte": min_confidence}}
        if context_filter:
            filters["context"] = context_filter

        skill_claims = self.db.query(filters)

        # Aggregate by value
        skill_data = defaultdict(
            lambda: {"claims": [], "contexts": set(), "confidences": []}
        )

        for claim in skill_claims:
            skill_data[claim.value]["claims"].append(claim)
            skill_data[claim.value]["contexts"].add(claim.context)
            skill_data[claim.value]["confidences"].append(claim.confidence)

        # Calculate stats
        results = []
        for skill, data in skill_data.items():
            results.append(
                {
                    "value": skill,
                    "count": len(data["claims"]),
                    "avg_confidence": sum(data["confidences"])
                    / len(data["confidences"]),
                    "contexts": sorted(list(data["contexts"])),
                    "evidence_count": len(data["claims"]),
                }
            )

        # Sort by count (primary) and avg_confidence (secondary)
        results.sort(key=lambda x: (x["count"], x["avg_confidence"]), reverse=True)

        return results[:limit]

    def get_skill_progression(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get timeline of skill acquisition.

        Returns:
            Dict mapping date -> list of skills first appearing on that date
        """
        skill_claims = self.db.get_by_type("skill")

        # Track first appearance of each skill
        skill_first_seen = {}

        for claim in skill_claims:
            if claim.value not in skill_first_seen:
                skill_first_seen[claim.value] = claim.document_date
            else:
                # Keep earliest date
                if claim.document_date < skill_first_seen[claim.value]:
                    skill_first_seen[claim.value] = claim.document_date

        # Group by date
        timeline = defaultdict(list)
        for skill, date in skill_first_seen.items():
            timeline[date].append(skill)

        # Sort dates
        return dict(sorted(timeline.items()))

    def get_context_breakdown(
        self, claim_type: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get breakdown of claims by context.

        Args:
            claim_type: Optional filter by claim type

        Returns:
            Dict mapping context -> stats
        """
        if claim_type:
            claims = self.db.get_by_type(claim_type)
        else:
            claims = list(self.db.claims.values())

        context_data = defaultdict(
            lambda: {
                "count": 0,
                "claim_types": Counter(),
                "avg_confidence": [],
                "unique_values": set(),
            }
        )

        for claim in claims:
            context_data[claim.context]["count"] += 1
            context_data[claim.context]["claim_types"][claim.claim_type] += 1
            context_data[claim.context]["avg_confidence"].append(claim.confidence)
            context_data[claim.context]["unique_values"].add(claim.value)

        # Calculate averages
        results = {}
        for context, data in context_data.items():
            results[context] = {
                "count": data["count"],
                "claim_types": dict(data["claim_types"]),
                "avg_confidence": sum(data["avg_confidence"])
                / len(data["avg_confidence"]),
                "unique_values": len(data["unique_values"]),
            }

        return results

    def get_missing_skills(self) -> Dict[str, List[str]]:
        """
        Find skills present in research/academic but missing from resumes.

        Returns:
            Dict with presentInPapers and missingFromResume
        """
        # Get skills by document type (inferred from filename patterns)
        all_skills = self.db.get_by_type("skill")

        paper_skills = set()
        resume_skills = set()

        for claim in all_skills:
            filename = claim.evidence.filename.lower()

            # Classify by filename
            if "research" in filename or "paper" in filename or ".tex" in filename:
                paper_skills.add(claim.value)
            elif "resume" in filename or "cv" in filename:
                resume_skills.add(claim.value)

        # Find gaps
        missing = paper_skills - resume_skills

        return {
            "presentInPapers": sorted(list(paper_skills)),
            "missingFromResume": sorted(list(missing)),
            "inResume": sorted(list(resume_skills)),
        }

    def get_value_profile(self, min_confidence: float = 0.7) -> List[Dict[str, Any]]:
        """
        Get inferred values from claims.

        Args:
            min_confidence: Minimum confidence threshold

        Returns:
            List of value dicts with frequency and evidence
        """
        value_claims = self.db.query(
            {"claim_type": "value", "confidence": {"$gte": min_confidence}}
        )

        # Count frequency
        value_counts = Counter([claim.value for claim in value_claims])

        results = []
        for value, count in value_counts.most_common():
            results.append(
                {"value": value, "frequency": count, "evidence_count": count}
            )

        return results

    def get_achievements_by_impact(self) -> List[Dict[str, Any]]:
        """
        Get achievements sorted by confidence (proxy for impact).

        Returns:
            List of achievement dicts
        """
        achievements = self.db.get_by_type("achievement")

        # Sort by confidence (explicit achievements ranked higher)
        achievements.sort(key=lambda x: x.confidence, reverse=True)

        results = []
        for claim in achievements:
            results.append(
                {
                    "value": claim.value,
                    "confidence": claim.confidence,
                    "context": claim.context,
                    "date": claim.document_date,
                    "source": f"{claim.evidence.filename} (p.{claim.evidence.page})",
                }
            )

        return results

    def get_experience_timeline(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get timeline of experiences.

        Returns:
            Dict mapping date -> list of experiences
        """
        experiences = self.db.get_by_type("experience")

        timeline = defaultdict(list)

        for claim in experiences:
            timeline[claim.document_date].append(
                {
                    "value": claim.value,
                    "context": claim.context,
                    "confidence": claim.confidence,
                    "source": claim.evidence.filename,
                }
            )

        return dict(sorted(timeline.items()))

    def get_growth_metrics(self) -> Dict[str, Any]:
        """
        Calculate growth metrics across time.

        Returns:
            Dict with various growth statistics
        """
        timeline = self.db.get_timeline()

        if not timeline:
            return {"error": "No timeline data available"}

        # Calculate metrics
        dates = sorted(timeline.keys())

        # Skills over time
        cumulative_skills = set()
        skills_by_date = {}

        for date in dates:
            date_skills = [
                claim.value for claim in timeline[date] if claim.claim_type == "skill"
            ]
            cumulative_skills.update(date_skills)
            skills_by_date[date] = {
                "new_skills": len(date_skills),
                "cumulative": len(cumulative_skills),
            }

        return {
            "date_range": {"earliest": dates[0], "latest": dates[-1]},
            "skills_progression": skills_by_date,
            "total_growth": len(cumulative_skills),
        }

    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate complete summary report.

        Returns:
            Comprehensive summary of all insights
        """
        return {
            "database_stats": self.db.get_stats(),
            "top_skills": self.get_top_skills(limit=10),
            "context_breakdown": self.get_context_breakdown(),
            "missing_skills": self.get_missing_skills(),
            "value_profile": self.get_value_profile(),
            "growth_metrics": self.get_growth_metrics(),
        }


if __name__ == "__main__":
    """Generate insights report from claims database."""
    import json
    from ...utils import get_project_root

    project_root = get_project_root()
    db_path = project_root / "cache" / "claims.json"

    if not db_path.exists():
        print(f"Error: Claims database not found at {db_path}")
        print("Run extractor.py first to generate claims.")
        exit(1)

    # Generate insights
    engine = InsightEngine(str(db_path))
    report = engine.generate_summary_report()

    # Save report
    report_path = project_root / "cache" / "insights_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print("Insights Report Generated")
    print(f"{'='*60}\n")
    print(f"Saved to: {report_path}\n")

    # Print summary
    print("Summary:")
    print(f"  Total Claims: {report['database_stats']['total_claims']}")
    print(f"  Top 5 Skills: {[s['value'] for s in report['top_skills'][:5]]}")
    print(
        f"  Missing from Resume: {len(report['missing_skills']['missingFromResume'])} skills"
    )
