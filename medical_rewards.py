"""Reward functions for medical reasoning RL (TRL GRPOTrainer compatible).

Each reward function takes ``completions`` (a list of strings, or of chat-message lists as
TRL passes them) plus dataset columns as keyword arguments, and returns one float per completion.

Expected output format of the model:

    <think>
    step-by-step reasoning
    </think>

    final answer / diagnosis

Dataset columns used as kwargs: ``question`` (str) and ``reference`` (the gold ``Response``).

Reward design
-------------
format_reward       [0, 1]   well-formed <think>...</think> followed by a non-empty answer
accuracy_reward     [0, 1]   multiple-choice: exact match on the option (verifiable);
                             free text: F1 over answer-specific terms vs. the reference
repetition_penalty  [-1, 0]  degenerate repetition / runaway think block
"""
import re
from collections import Counter

# ----------------------------------------------------------------------------- parsing

_THINK_OPEN, _THINK_CLOSE = "<think>", "</think>"


def completion_text(completion):
    """TRL passes chat completions as [{"role": "assistant", "content": ...}]; accept both."""
    if isinstance(completion, str):
        return completion
    return completion[-1]["content"]


def split_completion(text):
    """Return (think, answer, well_formed).

    well_formed: exactly one <think> at the very start, exactly one </think> after it, and
    no stray tags in the answer. If </think> is missing (e.g. truncated) the answer is "".
    If no tags are present at all, the whole text is treated as the answer.
    """
    n_open, n_close = text.count(_THINK_OPEN), text.count(_THINK_CLOSE)
    if n_open == 0 and n_close == 0:
        return "", text.strip(), False
    if n_close == 0:
        return text.split(_THINK_OPEN, 1)[-1].strip(), "", False

    head, _, answer = text.partition(_THINK_CLOSE)
    think = head.split(_THINK_OPEN, 1)[-1].strip()
    well_formed = (
        n_open == 1 and n_close == 1
        and head.lstrip().startswith(_THINK_OPEN)
        and bool(answer.strip())
    )
    return think, answer.strip(), well_formed


# ----------------------------------------------------------------------------- format

MIN_THINK_WORDS = 20


def format_reward(completions, **kwargs):
    """0.4 well-formed structure + 0.3 non-trivial reasoning + 0.3 non-empty clean answer."""
    rewards = []
    for c in completions:
        think, answer, well_formed = split_completion(completion_text(c))
        r = 0.0
        if well_formed:
            r += 0.4
        if len(think.split()) >= MIN_THINK_WORDS:
            r += 0.3
        if answer and _THINK_OPEN not in answer and _THINK_CLOSE not in answer:
            r += 0.3
        rewards.append(r)
    return rewards


# ----------------------------------------------------------------------------- options (MCQ)

_OPTION_LINE = re.compile(r"(?m)^\s*\(?([A-E])[\.\):]\s+(.+?)\s*$")
_LEADING_LETTER = re.compile(r"^\s*(?:(?:the\s+)?(?:correct\s+)?(?:answer|option|choice)\s*(?:is|:)?\s*)?\(?([A-E])[\.\):,]?(?:\s|$)", re.I)
_ANSWER_IS_LETTER = re.compile(r"\b(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-E])\b(?![a-z'])", re.I)


def _norm(s):
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()


def parse_options(question):
    """{'A': 'text', ...} if the question is multiple choice (>= 3 options incl. A and B)."""
    opts = {m.group(1): m.group(2) for m in _OPTION_LINE.finditer(question)}
    return opts if len(opts) >= 3 and {"A", "B"} <= opts.keys() else {}


def _earliest_option_mention(text, options):
    """Option whose text is mentioned earliest in ``text`` (None if no option is mentioned)."""
    t, best = " " + _norm(text) + " ", None
    for letter, opt in options.items():
        o = _norm(opt)
        pos = t.find(" " + o + " ") if o else -1
        if pos >= 0 and (best is None or pos < best[0]):
            best = (pos, letter)
    return best[1] if best else None


def extract_option(answer, options):
    """Which option a model answer picks: explicit letter first, else earliest option text."""
    m = _LEADING_LETTER.match(answer) or _ANSWER_IS_LETTER.search(answer)
    if m and m.group(1).upper() in options:
        return m.group(1).upper()
    return _earliest_option_mention(answer, options)


def gold_option(question, reference):
    """Recover the gold option letter from the free-text reference; None if not resolvable."""
    options = parse_options(question)
    if not options:
        return None
    m = _LEADING_LETTER.match(reference)
    if m and m.group(1).upper() in options:
        return m.group(1).upper()
    # the reference restates the option text; only trust it when exactly one option appears
    t = " " + _norm(reference) + " "
    hits = [k for k, v in options.items() if _norm(v) and " " + _norm(v) + " " in t]
    return hits[0] if len(hits) == 1 else None


# ----------------------------------------------------------------------------- free-text accuracy

_STOP = set("""
a an the of and or to in on at for with by from as is are was were be been being it its this that these those
he she they his her their them we you your i not no yes but if then than so such can could would should may
might will shall do does did done has have had having which who whom whose what when where why how there here
also more most much many some any all each both other another only just very too into over under between
about after before during while because due through up down out off again further once further than
likely based given suggests suggest indicates indicate typically commonly usually often generally
condition patient patients case scenario described description following question answer
therefore however thus hence overall consider considering considered particularly especially specific
""".split())
_WORD = re.compile(r"[a-z0-9][a-z0-9\-']*")


def _stem(w):
    for suf in ("ies", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def content_terms(text):
    return {_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1}


def answer_terms(answer, question):
    """Terms the answer introduces beyond the question: the actual diagnosis/mechanism/answer."""
    return content_terms(answer) - content_terms(question)


def _f1(pred, gold):
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    return 2 * p * r / (p + r)


# A correct answer worded differently from the reference only reaches F1 ~0.25 (median, using the
# last 90 words of Complex_CoT as a stand-in for a concise correct answer), so rescale by 1/0.5
# to use more of the [0, 1] range. Side effect: once F1 >= 0.5 extra terms cost nothing.
F1_SATURATION = 0.5


def free_text_score(answer, question, reference):
    gold = answer_terms(reference, question) or content_terms(reference)
    pred = answer_terms(answer, question)
    return min(1.0, _f1(pred, gold) / F1_SATURATION)


def accuracy_reward(completions, question=None, reference=None, **kwargs):
    """Verifiable exact match for multiple-choice; soft answer-term F1 otherwise.

    The score is computed on the text after </think> only, so reasoning cannot leak credit.
    """
    rewards = []
    for c, q, ref in zip(completions, question, reference):
        _, answer, _ = split_completion(completion_text(c))
        if not answer:
            rewards.append(0.0)
            continue
        gold = gold_option(q, ref)
        if gold is not None:
            rewards.append(1.0 if extract_option(answer, parse_options(q)) == gold else 0.0)
        else:
            rewards.append(free_text_score(answer, q, ref))
    return rewards


# ----------------------------------------------------------------------------- repetition / length

MAX_THINK_WORDS = 600


def _repeat_ratio(text, n=4):
    words = text.lower().split()
    if len(words) < n + 1:
        return 0.0
    grams = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    return 1.0 - len(grams) / sum(grams.values())


def repetition_penalty(completions, **kwargs):
    """Up to -0.5 for repeated 4-grams (loops) and up to -0.5 for an over-long think block."""
    rewards = []
    for c in completions:
        text = completion_text(c)
        think, _, _ = split_completion(text)
        pen = -min(0.5, max(0.0, _repeat_ratio(text) - 0.2))
        over = len(think.split()) - MAX_THINK_WORDS
        pen -= min(0.5, max(0, over) / MAX_THINK_WORDS)
        rewards.append(pen)
    return rewards


# ----------------------------------------------------------------------------- combination

REWARD_FUNCS = [format_reward, accuracy_reward, repetition_penalty]
REWARD_WEIGHTS = [0.2, 0.7, 0.1]  # for GRPOConfig(reward_weights=...)


def total_reward(completions, **kwargs):
    """Weighted sum, for inspection outside the trainer."""
    parts = [f(completions, **kwargs) for f in REWARD_FUNCS]
    return [sum(w * p[i] for w, p in zip(REWARD_WEIGHTS, parts)) for i in range(len(completions))]
