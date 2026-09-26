"""LlamaIndex orchestration for Portfolio corpus generation."""

from genai_template.workflow.portfolio.workflow.corpus import (
    PortfolioCorpusWorkflow,
    run_portfolio_corpus_workflow,
)

__all__ = ["PortfolioCorpusWorkflow", "run_portfolio_corpus_workflow"]
