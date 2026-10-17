"""
Pipeline orchestrator — wires all standalone modules into a single
question-to-answer flow.

Flow:
  0. Injection Defense  (blocks attacks)
  1. Auto-Memory        (inject recalled facts)
  2. Schema Retrieval   (get relevant tables/columns)
  3. Glossary + Corrections (feed into prompt)
  4. SQL Generator + Critic + Reasoning Engine (retry)
  5. Executor           (run SQL read-only)
  6. Result Validator   (sanity + assumptions)
  7. Confidence Gate    (final trust decision)
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from src.agent.uncertainty import UncertaintyDecomposer, UncertaintyDecomposition, TargetedClarification
from src.agent.answer_set import AnswerSetGenerator, HonestAnswerSet
from src.agent.counterfactual import CounterfactualVerifier, CounterfactualResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Complete result from the pipeline."""
    question: str
    sql: str | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    column_names: list[str] = field(default_factory=list)
    success: bool = False
    refused: bool = False
    refusal_reason: str = ""
    error: str | None = None
    confidence_level: str = ""
    confidence_score: float = 0.0
    validation_findings: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    cross_check_sql: str | None = None
    attempts: int = 1
    dialect: str = "sqlite"
    recalled_facts: list[str] = field(default_factory=list)
    glossary_terms: list[str] = field(default_factory=list)
    correction_rules: list[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    uncertainty: UncertaintyDecomposition | None = None
    targeted_clarification: TargetedClarification | None = None
    answer_set: HonestAnswerSet | None = None
    counterfactual_result: CounterfactualResult | None = None

    @property
    def has_warnings(self) -> bool:
        return len(self.validation_findings) > 0

    @property
    def has_assumptions(self) -> bool:
        return len(self.assumptions) > 0

    def format_response(self) -> str:
        parts = []
        if self.refused:
            parts.append(f"❌ Refused: {self.refusal_reason}")
            if self.targeted_clarification:
                parts.append(f"\n💡 Clarification Needed: {self.targeted_clarification.question}")
                if self.targeted_clarification.options:
                    parts.append("  Options: " + ", ".join(self.targeted_clarification.options))
            return "\n".join(parts)
        if self.error:
            parts.append(f"❌ Error: {self.error}")
            return "\n".join(parts)
        if self.rows:
            parts.append(f"✅ Query returned {self.row_count} row(s)")
        if self.confidence_level:
            parts.append(f"📊 Confidence: {self.confidence_level}")
        if self.uncertainty:
            parts.append(f"🎯 Uncertainty: {self.uncertainty.composite_uncertainty:.0%} ({self.uncertainty.dominant_dimension})")
        if self.answer_set and self.answer_set.has_alternatives:
            parts.append("\n" + self.answer_set.format_display())
        if self.targeted_clarification and not self.refused:
            parts.append(f"\n💡 Note: {self.targeted_clarification.question}")
        if self.validation_findings:
            parts.append("\n### Validation Findings")
            for f in self.validation_findings:
                parts.append(f"  {f}")
        if self.assumptions:
            parts.append("\n### Assumptions Made")
            for a in self.assumptions:
                parts.append(f"  📋 {a}")
        return "\n".join(parts)


class Pipeline:
    """
    Orchestrates the full question-to-answer flow.

    Required modules: sql_generator, executor, critic.
    Optional: schema_index, reasoning_engine, result_validator,
              confidence_scorer, injection_defense, glossary_store,
              correction_memory, auto_memory,
              uncertainty_decomposer, counterfactual_verifier, answer_set_generator.
    """

    def __init__(
        self,
        sql_generator,
        executor,
        critic,
        schema_index=None,
        reasoning_engine=None,
        result_validator=None,
        confidence_scorer=None,
        injection_defense=None,
        glossary_store=None,
        correction_memory=None,
        auto_memory=None,
        uncertainty_decomposer=None,
        counterfactual_verifier=None,
        answer_set_generator=None,
        dialect: str = "sqlite",
    ):
        if sql_generator is None:
            raise ValueError("sql_generator is required")
        if executor is None:
            raise ValueError("executor is required")
        if critic is None:
            raise ValueError("critic is required")

        self._sql_generator = sql_generator
        self._executor = executor
        self._critic = critic
        self._schema_index = schema_index
        self._reasoning_engine = reasoning_engine
        self._result_validator = result_validator
        self._confidence_scorer = confidence_scorer
        self._injection_defense = injection_defense
        self._glossary_store = glossary_store
        self._correction_memory = correction_memory
        self._auto_memory = auto_memory
        self._uncertainty_decomposer = uncertainty_decomposer or UncertaintyDecomposer()
        self._counterfactual_verifier = counterfactual_verifier or CounterfactualVerifier(executor=executor)
        self._answer_set_generator = answer_set_generator or AnswerSetGenerator()
        self._dialect = dialect

    def run(self, question: str) -> PipelineResult:
        """Run the full pipeline for a user question."""
        result = PipelineResult(question=question, dialect=self._dialect)

        # Stage 0: Injection Defense
        if self._injection_defense:
            try:
                scan = self._injection_defense.scan_text(question)
                if hasattr(scan, 'threat_level'):
                    from src.agent.injection_defense import ThreatLevel
                    if scan.threat_level == ThreatLevel.BLOCKED:
                        result.refused = True
                        result.refusal_reason = "Question contains suspected prompt injection and was blocked."
                        return result
            except Exception as e:
                logger.error(f"Injection defense error (continuing): {e}")

        # Stage 1: Auto-Memory Recall
        recalled_facts: list[str] = []
        if self._auto_memory:
            try:
                relevant = self._auto_memory.prefetch(question)
                recalled_facts = [f.content for f in relevant]
                result.recalled_facts = recalled_facts
            except Exception as e:
                logger.error(f"Auto-memory recall error (continuing): {e}")

        # Stage 2: Schema Retrieval
        schema_context = ""
        if self._schema_index:
            try:
                schema_context = self._schema_index.get_schema_context(question)
            except Exception as e:
                logger.error(f"Schema retrieval error (continuing): {e}")

        # Stage 3: Glossary + Correction Rules
        glossary_context = ""
        if self._glossary_store:
            try:
                matched_terms = self._glossary_store.lookup(question)
                glossary_context = self._glossary_store.format_for_prompt(matched_terms)
                result.glossary_terms = [t.term for t in matched_terms]
            except Exception as e:
                logger.error(f"Glossary error (continuing): {e}")

        correction_rules: list[str] = []
        if self._correction_memory:
            try:
                correction_rules = self._correction_memory.get_prompt_rules(question)
                result.correction_rules = correction_rules
            except Exception as e:
                logger.error(f"Correction memory error (continuing): {e}")

        if recalled_facts:
            facts_section = "\n\nRecalled facts:\n"
            for fact in recalled_facts:
                facts_section += f"- {fact}\n"
            schema_context += facts_section

        # Stage 4: SQL Generation + Critic + Retry
        if self._reasoning_engine:
            try:
                def generate_fn(question, schema, retry_context=None, **kwargs):
                    gen_result = self._sql_generator.generate(
                        question=question, schema_context=schema,
                        glossary_context=glossary_context,
                        correction_rules=correction_rules,
                    )
                    return gen_result.sql

                engine_result = self._reasoning_engine.run(
                    question=question, schema=schema_context,
                    generate_fn=generate_fn,
                )
                result.sql = engine_result.sql
                result.attempts = engine_result.total_attempts
                if not engine_result.success:
                    result.error = engine_result.final_diagnosis
                    return result
            except Exception as e:
                logger.error(f"Reasoning engine error: {e}")
                result.error = str(e)
                return result
        else:
            try:
                gen_result = self._sql_generator.generate(
                    question=question, schema_context=schema_context,
                    glossary_context=glossary_context,
                    correction_rules=correction_rules,
                )
                result.sql = gen_result.sql
                if result.sql:
                    critique = self._critic.evaluate(result.sql)
                    if not critique.passed:
                        result.error = critique.diagnosis
                        return result
            except Exception as e:
                logger.error(f"SQL generation error: {e}")
                result.error = str(e)
                return result

        # Stage 4.5: Uncertainty Decomposition & Targeted Clarification
        if self._uncertainty_decomposer and result.sql:
            try:
                schema_tables = {}
                foreign_keys = []
                if self._schema_index and hasattr(self._schema_index, "get_tables"):
                    try:
                        schema_tables = self._schema_index.get_tables()
                    except Exception:
                        pass
                unc = self._uncertainty_decomposer.decompose(
                    question=question,
                    sql=result.sql,
                    schema_tables=schema_tables,
                    foreign_keys=foreign_keys,
                )
                result.uncertainty = unc
                if unc.needs_clarification and unc.clarification:
                    result.targeted_clarification = unc.clarification
            except Exception as e:
                logger.error(f"Uncertainty decomposition error (continuing): {e}")

        # Stage 4.6: Honest Answer Set (Ambiguity / Warning Band Check)
        if self._answer_set_generator and result.sql:
            try:
                if self._answer_set_generator.should_generate_answer_set(
                    confidence_score=0.7,
                    uncertainty=result.uncertainty,
                ):
                    answer_set = self._answer_set_generator.build_answer_set(
                        question=question,
                        primary_sql=result.sql,
                        uncertainty=result.uncertainty,
                        executor=self._executor,
                    )
                    if answer_set.has_alternatives:
                        result.answer_set = answer_set
            except Exception as e:
                logger.error(f"Answer set generation error (continuing): {e}")

        # Stage 5: Execution
        if result.sql:
            try:
                exec_result = self._executor.execute(result.sql)
                if exec_result.error:
                    critique = self._critic.evaluate(result.sql, execution_error=exec_result.error)
                    result.error = critique.diagnosis
                    return result
                result.rows = exec_result.rows
                result.row_count = exec_result.row_count
                result.column_names = exec_result.column_names
                result.execution_time_ms = exec_result.execution_time_ms
                result.success = True
            except Exception as e:
                logger.error(f"Execution error: {e}")
                result.error = str(e)
                return result

        # Stage 5.5: Counterfactual Verification (Gated Execution + Safe Read-Only)
        if self._counterfactual_verifier and result.sql and result.success:
            try:
                unc_score = result.uncertainty.composite_uncertainty if result.uncertainty else 0.0
                cf_result = self._counterfactual_verifier.verify(
                    sql=result.sql,
                    original_rows=result.rows,
                    confidence_score=1.0,
                    uncertainty_score=unc_score,
                )
                result.counterfactual_result = cf_result
                if not cf_result.passed:
                    for finding in cf_result.findings:
                        result.validation_findings.append(f"Counterfactual finding: {finding}")
            except Exception as e:
                logger.error(f"Counterfactual verification error (continuing): {e}")

        # Stage 6: Result Validation
        if self._result_validator and result.success and result.rows:
            try:
                validation = self._result_validator.validate(
                    sql=result.sql or "", question=question,
                    rows=result.rows, column_names=result.column_names,
                    row_count=result.row_count,
                )
                result.validation_findings.extend([f.display() for f in validation.findings])
                result.assumptions.extend([a.description for a in validation.assumptions])
                result.cross_check_sql = validation.cross_check_sql
                if not validation.is_trustworthy:
                    result.success = False
                    result.error = "Result validation failed — data may be incorrect"
            except Exception as e:
                logger.error(f"Result validation error (continuing): {e}")

        # Stage 7: Confidence Gate
        if self._confidence_scorer and result.success:
            try:
                cf_passed = True
                if result.counterfactual_result:
                    cf_passed = result.counterfactual_result.passed
                unc_score = result.uncertainty.composite_uncertainty if result.uncertainty else 0.0

                conf = self._confidence_scorer.score(
                    result_sanity=0.0 if result.validation_findings else 1.0,
                    counterfactual_passed=cf_passed,
                    uncertainty_score=unc_score,
                )
                result.confidence_level = conf.display()
                result.confidence_score = conf.score
                if not conf.should_proceed():
                    result.refused = True
                    result.refusal_reason = conf.reason
            except Exception as e:
                logger.error(f"Confidence scoring error (continuing): {e}")

        return result
