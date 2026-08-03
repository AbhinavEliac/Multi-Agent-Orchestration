from pydantic import BaseModel


class LearnerSchema(BaseModel):

    title: str

    summary: str

    audience: str

    intent: str

    tone: str

    writing_style: str

    seo_intent: str

    primary_keywords: list[str]

    secondary_keywords: list[str]

    structure: list[str]


class PlannerSchema(BaseModel):

    strengths: list[str]

    weaknesses: list[str]

    missing_topics: list[str]

    seo_improvements: list[str]

    research_query: str

    missing_statistics: list[str]

    missing_examples: list[str]

    missing_sections: list[str]


class ResearchSchema(BaseModel):

    updated_facts: list[str]

    statistics: list[str]

    examples: list[str]

    seo_keywords: list[str]

    lsi_keywords: list[str]

    faqs: list[str]

    references: list[str]


class EvaluationSchema(BaseModel):

    language: int

    facts: int

    structure: int

    language_feedback: str

    facts_feedback: str

    structure_feedback: str