"""The atomic-claim verifier: decomposition, labelling judge, grading judge, score.

  from pgxverifier.verifier import Verifier
  v = Verifier()
  result = v.verify(question, answer, gt_claims=[...])        # curated GT claims
  result = v.verify(question, answer, ground_truth="...")     # or a GT text, decomposed here
  result["scores"]   # precision, recall, F1.5, claim counts

The answer is decomposed with the same two passes and deduplication as a GT text. GT claims
are normally decomposed once and curated, then passed in as gt_claims.
"""
import threading

from . import config
from .steps import decompose, grade, label, score


class Verifier:
    def __init__(self):
        self._decompose_prompts = decompose.load_prompts()
        self._label_prompt = (config.PROMPT_DIR / "label_judge.txt").read_text(encoding="utf-8")
        self._grade_fwd = (config.PROMPT_DIR / "grade_forward.txt").read_text(encoding="utf-8")
        self._grade_bwd = (config.PROMPT_DIR / "grade_backward.txt").read_text(encoding="utf-8")
        self.decomposer = decompose.Endpoint()
        self._labeller = None          # created on first use: it needs XAI_API_KEY
        self._grader = None
        self._lock = threading.Lock()  # verify() may run in several threads

    @property
    def labeller(self):
        with self._lock:
            if self._labeller is None:
                self._labeller = label.GrokClient()
        return self._labeller

    @property
    def grader(self):
        with self._lock:
            if self._grader is None:
                self._grader = grade.make_grader()
        return self._grader

    def claims(self, question, text):
        """A text -> {"claims": [...], plus the record of the two passes and the dedup}."""
        return decompose.decompose(self.decomposer, self._decompose_prompts, question, text)

    def verify(self, question, answer, gt_claims=None, ground_truth=None):
        if gt_claims is None:
            if not ground_truth:
                raise ValueError("give gt_claims or ground_truth")
            gt_record = self.claims(question, ground_truth)
            gt_claims = gt_record["claims"]
            if len(gt_claims) < 3:
                gt_record["warnings"].append(f"GT decomposed to only {len(gt_claims)} claims "
                                             f"(possibly under-extracted or genuinely sparse)")
        else:
            gt_claims = [str(c) for c in gt_claims]
            gt_record = None
        ans_record = self.claims(question, answer)
        ans_claims = ans_record["claims"]
        labels = label.judge(self.labeller, self._label_prompt, question, ans_claims, gt_claims)
        credits = grade.grade_pair(self.grader, self._grade_fwd, self._grade_bwd, question,
                                   ans_claims, gt_claims, labels)
        return {"scores": score.score(labels, credits, len(ans_claims), len(gt_claims)),
                "answer_claims": ans_claims, "gt_claims": gt_claims,
                "labels": labels, "credits": credits,
                "answer_decomposition": ans_record, "gt_decomposition": gt_record}


_default = None


def verify(question, answer, gt_claims=None, ground_truth=None):
    """Verify one answer with a shared Verifier."""
    global _default
    if _default is None:
        _default = Verifier()
    return _default.verify(question, answer, gt_claims=gt_claims, ground_truth=ground_truth)
