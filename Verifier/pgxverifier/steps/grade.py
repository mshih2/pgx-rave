"""Grading judge (Mistral-Small-24B): graded credit for every positively labelled claim.

Each answer claim labelled addressed / partially_addressed and each GT claim labelled covered /
partial is re-read on its own, one call per claim, on a three-grade scale. The credit is read
from the first grade token's log-probabilities, not sampled:

  c = p(full) + 0.5 * p(partial),  renormalised over the three grade tokens

Claims labelled extra, wrong or missing are not sent and score 0. A claim whose log-probabilities
cannot be read keeps its labelling-judge weight (1.0 full, 0.5 partial).
"""
import math
from concurrent.futures import ThreadPoolExecutor

from .. import config

# Grade token prefix -> grade. Matched on the first token that starts one of the grades.
FORWARD_GRADES = {"support": "full", "partial": "half", "wrong": "none"}
BACKWARD_GRADES = {"cover": "full", "partial": "half", "missing": "none"}
FWD_POSITIVE = ("addressed", "partially_addressed")
BWD_POSITIVE = ("covered", "partial")
FWD_HARD = {"addressed": 1.0, "partially_addressed": 0.5}
BWD_HARD = {"covered": 1.0, "partial": 0.5}


def _numbered(claims):
    return "\n".join(f"[{i}] {c}" for i, c in enumerate(claims))


def credit_from_logprobs(content, grades):
    """Expected credit at the first grade token, or None if no token starts a grade."""
    for tok in content:
        t = tok.token.strip().strip('"').lower()
        if not any(t.startswith(k) for k in grades):
            continue
        p_full = p_half = total = 0.0
        for alt in tok.top_logprobs:
            a = alt.token.strip().strip('"').lower()
            hit = next((v for k, v in grades.items() if a.startswith(k)), None)
            if hit is None:
                continue
            p = math.exp(alt.logprob)
            total += p
            if hit == "full":
                p_full += p
            elif hit == "half":
                p_half += p
        return (p_full + 0.5 * p_half) / total if total > 0 else None
    return None


def make_grader(base_url=None, model=None):
    """-> grade(prompt, grades) backed by the grading judge's OpenAI-compatible server."""
    from openai import OpenAI
    cli = OpenAI(base_url=base_url or config.GRADER_URL, api_key="EMPTY",
                 timeout=config.GRADER_TIMEOUT)
    model = model or config.GRADER_MODEL

    def grade(prompt, grades):
        r = cli.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], temperature=0.0,
            max_tokens=config.GRADER_MAX_TOKENS, logprobs=True,
            top_logprobs=config.GRADER_TOP_LOGPROBS)
        return credit_from_logprobs(r.choices[0].logprobs.content, grades)

    return grade


def grade_pair(grader, fwd_template, bwd_template, question, answer_claims, gt_claims, labels,
               workers=None):
    """-> {"fwd_credit": {i: c}, "bwd_credit": {j: c}} for the positively labelled claims."""
    fwd_labels = {int(k): v for k, v in labels["fwd_labels"].items()}
    bwd_labels = {int(k): v for k, v in labels["bwd_labels"].items()}
    gt_block, ans_block = _numbered(gt_claims), _numbered(answer_claims)
    fwd_idx = [i for i in range(len(answer_claims)) if fwd_labels.get(i) in FWD_POSITIVE]
    bwd_idx = [j for j in range(len(gt_claims)) if bwd_labels.get(j) in BWD_POSITIVE]

    def do_fwd(i):
        p = (fwd_template.replace("{question_text}", question)
             .replace("{claim_text}", answer_claims[i])
             .replace("{numbered_gt_claims}", gt_block))
        c = grader(p, FORWARD_GRADES)
        return c if c is not None else FWD_HARD.get(fwd_labels.get(i), 0.0)

    def do_bwd(j):
        p = (bwd_template.replace("{question_text}", question)
             .replace("{claim_text}", gt_claims[j])
             .replace("{numbered_response_claims}", ans_block))
        c = grader(p, BACKWARD_GRADES)
        return c if c is not None else BWD_HARD.get(bwd_labels.get(j), 0.0)

    with ThreadPoolExecutor(max_workers=max(1, workers or config.WORKERS)) as ex:
        fwd_credits = list(ex.map(do_fwd, fwd_idx))
        bwd_credits = list(ex.map(do_bwd, bwd_idx))
    return {"fwd_credit": {str(i): round(c, 4) for i, c in zip(fwd_idx, fwd_credits)},
            "bwd_credit": {str(j): round(c, 4) for j, c in zip(bwd_idx, bwd_credits)}}
