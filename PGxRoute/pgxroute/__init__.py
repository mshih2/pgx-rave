"""PGxRoute answer pipelines (API-only, Reasoning-only, Full).

Entry point: `pgxroute.pipelines.answer(question, pipeline)`. Nothing is imported here, so the
ranking subprocess (`python -m pgxroute.steps.retrieval`) can run in an environment that has
FlagEmbedding but not the rest of the requirements.
"""
