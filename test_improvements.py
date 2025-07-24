#!/usr/bin/env python3
"""
Test script to validate the improvements made to NovaEval scorers.
"""

import logging
from io import StringIO

from novaeval.scorers.base import BaseScorer, validate_scorer_inputs
from novaeval.scorers.accuracy import AccuracyScorer
from novaeval.scorers.g_eval import GEvalScorer, GEvalCriteria


class TestScorer(BaseScorer):
    """Test scorer for validation."""
    
    def score(self, prediction, ground_truth, context=None):
        return 1.0 if prediction == ground_truth else 0.0


def test_input_validation():
    """Test the new centralized input validation."""
    print("Testing input validation improvements...")
    
    # Valid inputs
    is_valid, msg = validate_scorer_inputs("pred", "truth", {"key": "value"})
    assert is_valid, f"Expected valid inputs to pass: {msg}"
    
    # Invalid prediction type
    is_valid, msg = validate_scorer_inputs(123, "truth")
    assert not is_valid, "Expected invalid prediction type to fail"
    assert "string" in msg.lower()
    
    # Invalid context type  
    is_valid, msg = validate_scorer_inputs("pred", "truth", "invalid")
    assert not is_valid, "Expected invalid context type to fail"
    
    # Empty strings with allow_empty=False
    is_valid, msg = validate_scorer_inputs("", "truth", allow_empty=False)
    assert not is_valid, "Expected empty prediction to fail when not allowed"
    
    print("✓ Input validation tests passed")


def test_logging_improvements():
    """Test improved logging in batch scoring."""
    print("Testing logging improvements...")
    
    # Set up logging capture
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    logger = logging.getLogger('novaeval.scorers.base')
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    
    class FailingScorer(BaseScorer):
        def score(self, prediction, ground_truth, context=None):
            if "fail" in prediction:
                raise ValueError("Intentional failure")
            return 0.5
    
    scorer = FailingScorer("test_failing")
    
    # Test batch scoring with failures
    predictions = ["good1", "fail_me", "good2"]
    ground_truths = ["good1", "fail_me", "good2"]
    
    scores = scorer.score_batch(predictions, ground_truths)
    
    # Verify scores
    assert scores == [0.5, 0.0, 0.5], f"Expected [0.5, 0.0, 0.5], got {scores}"
    
    # Verify logging occurred
    log_output = log_capture.getvalue()
    assert "Scoring failed" in log_output, "Expected failure to be logged"
    assert "test_failing" in log_output, "Expected scorer name in log"
    
    logger.removeHandler(handler)
    print("✓ Logging improvements tests passed")


def test_memory_management():
    """Test memory management features."""
    print("Testing memory management improvements...")
    
    scorer = TestScorer("memory_test")
    
    # Add some scores
    for i in range(10):
        scorer._track_score(float(i))
    
    assert len(scorer.scores_history) == 10, "Expected 10 scores in history"
    
    # Test clearing with keep_recent
    scorer.clear_history(keep_recent=3)
    assert len(scorer.scores_history) == 3, "Expected 3 scores after clearing"
    assert scorer.scores_history == [7.0, 8.0, 9.0], "Expected last 3 scores"
    
    # Test clearing all
    scorer.clear_history()
    assert len(scorer.scores_history) == 0, "Expected empty history after clear all"
    
    print("✓ Memory management tests passed")


def test_batch_validation():
    """Test improved batch scoring validation."""
    print("Testing batch scoring validation...")
    
    scorer = TestScorer("batch_test")
    
    # Test mismatched lengths
    try:
        scorer.score_batch(["a", "b"], ["x"])  # Different lengths
        assert False, "Expected ValueError for mismatched lengths"
    except ValueError as e:
        assert "Mismatched lengths" in str(e)
    
    # Test mismatched context lengths
    try:
        scorer.score_batch(["a", "b"], ["x", "y"], [{"k": "v"}])  # 1 context for 2 items
        assert False, "Expected ValueError for mismatched context lengths"
    except ValueError as e:
        assert "Mismatched lengths" in str(e)
    
    print("✓ Batch validation tests passed")


def test_answer_extraction_optimization():
    """Test optimized answer extraction in AccuracyScorer."""
    print("Testing answer extraction optimization...")
    
    scorer = AccuracyScorer()
    
    # Test various patterns
    test_cases = [
        ("Answer: B", "B"),
        ("The answer is C", "C"),
        ("**A.** This is correct", "A"),
        ("D", "D"),  # Single letter
        ("Some text with B at the end", "B"),
    ]
    
    for prediction, expected in test_cases:
        extracted = scorer._extract_answer(prediction)
        assert extracted.upper() == expected, f"Expected {expected}, got {extracted} for '{prediction}'"
    
    print("✓ Answer extraction optimization tests passed")


def test_g_eval_parsing_robustness():
    """Test improved G-Eval response parsing."""
    print("Testing G-Eval parsing robustness...")
    
    criteria = GEvalCriteria(
        name="Test",
        description="Test criteria",
        steps=["Step 1", "Step 2"]
    )
    
    class MockModel:
        async def generate(self, prompt):
            return "Response without clear score format"
    
    class TestGEvalScorer(GEvalScorer):
        def score(self, prediction, ground_truth, context=None):
            # Simple implementation for testing
            return 0.5
    
    scorer = TestGEvalScorer(MockModel(), criteria)
    
    # Test robust parsing with no clear score
    score, reasoning = scorer._parse_response("This is a response with score buried 3 somewhere")
    assert score > 0, "Expected to extract some score even from malformed response"
    
    # Test fraction parsing
    score, reasoning = scorer._parse_response("Score: 4/5")
    assert 0.75 <= score <= 0.8, f"Expected ~0.75-0.8 for 4/5, got {score}"
    
    print("✓ G-Eval parsing robustness tests passed")


def main():
    """Run all improvement tests."""
    print("Running NovaEval Scorer Improvements Test Suite")
    print("=" * 50)
    
    test_input_validation()
    test_logging_improvements()
    test_memory_management()
    test_batch_validation()
    test_answer_extraction_optimization()
    test_g_eval_parsing_robustness()
    
    print("=" * 50)
    print("🎉 All improvement tests passed!")


if __name__ == "__main__":
    main()