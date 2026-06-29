"""
Zero-shot prompt templates for author profiling experiments.
All prompts use {TEXT} as the placeholder for the input text.

Modes:
  joint    — one prompt predicts both gender AND age
  separate — dedicated prompt per task (gender-only or age-only)
"""

GENDER_LABELS = ["F", "M"]
AGE_LABELS = ["0-25", "26-35", "36-65", "66-"]

# JOINT PROMPTS  (predict gender + age together)

JOINT_BASELINE = """
You are an expert in linguistic profiling.

Predict the gender and age group of the author of the following text.

Allowed labels:

Gender:
M
F

Age:
0-25
26-35
36-65
66-

Rules:
- Choose exactly one gender and one age group.
- If uncertain, choose the most likely option.
- Output only in the required format, nothing else.

Output format:
Gender: <M or F>
Age: <0-25 | 26-35 | 36-65 | 66->

Text:
\"\"\"
{TEXT}
\"\"\"
""".strip()

JOINT_STYLE = """
You are a linguistic analyst specialized in author profiling.

Determine the author's gender and age group based ONLY on writing style cues.

Focus on:
- vocabulary and word choice
- use of slang or formal language
- punctuation and capitalization
- sentence length and structure
- tone and emotional expression
- degree of formality

Allowed labels:

Gender:
M
F

Age:
0-25
26-35
36-65
66-

Rules:
- Choose exactly one gender and one age group.
- If uncertain, choose the most likely option.
- Do not explain your reasoning.
- Output only in the required format, nothing else.

Output format:
Gender: <M or F>
Age: <0-25 | 26-35 | 36-65 | 66->

Text:
\"\"\"
{TEXT}
\"\"\"
""".strip()

JOINT_DEMOGRAPHIC = """
You are a linguistic analyst specialized in demographic author profiling.

Predict the gender and age group of the author based on linguistic patterns.

Demographic markers to consider:

Gender:
- M: direct assertions, less hedging, more aggressive or blunt language, fewer qualifiers
- F: hedging phrases ("I think", "maybe", "perhaps"),
  more emotional expression, more qualifiers and politeness markers

Age:
- 0-25: casual language, internet slang, abbreviations,
  short sentences, informal tone, minimal punctuation
- 26-35: mix of formal and informal, complete sentences, references to work or adult life
- 36-65: more formal vocabulary, longer sentences, traditional expressions, less slang
- 66-: very formal language, old-fashioned expressions,
  avoids slang, longer and more complex sentences

Allowed labels:

Gender:
M
F

Age:
0-25
26-35
36-65
66-

Rules:
- Choose exactly one gender and one age group.
- If uncertain, choose the most likely option.
- Do not explain your reasoning.
- Output only in the required format, nothing else.

Output format:
Gender: <M or F>
Age: <0-25 | 26-35 | 36-65 | 66->

Text:
\"\"\"
{TEXT}
\"\"\"
""".strip()

# SEPARATE GENDER PROMPTS  (predict gender only)

SEPARATE_GENDER_BASELINE = """
You are an expert in linguistic profiling.

Predict the gender of the author of the following text.

Allowed labels:
M
F

Rules:
- Choose exactly one label.
- If uncertain, choose the most likely option.
- Output only in the required format, nothing else.

Output format:
Gender: <M or F>

Text:
\"\"\"
{TEXT}
\"\"\"
""".strip()

SEPARATE_GENDER_STYLE = """
You are a linguistic analyst specialized in author profiling.

Determine the author's gender based ONLY on writing style cues.

Focus on:
- vocabulary and word choice
- use of slang or formal language
- punctuation and capitalization
- sentence length and structure
- tone and emotional expression
- degree of formality

Allowed labels:
M
F

Rules:
- Choose exactly one label.
- If uncertain, choose the most likely option.
- Do not explain your reasoning.
- Output only in the required format, nothing else.

Output format:
Gender: <M or F>

Text:
\"\"\"
{TEXT}
\"\"\"
""".strip()

SEPARATE_GENDER_DEMOGRAPHIC = """
You are a linguistic analyst specialized in demographic author profiling.

Predict the gender of the author based on linguistic patterns.

Gender markers to consider:
- M: direct assertions, less hedging, more aggressive or blunt language, fewer qualifiers
- F: hedging phrases ("I think", "maybe", "perhaps"),
  more emotional expression, more qualifiers and politeness markers

Allowed labels:
M
F

Rules:
- Choose exactly one label.
- If uncertain, choose the most likely option.
- Do not explain your reasoning.
- Output only in the required format, nothing else.

Output format:
Gender: <M or F>

Text:
\"\"\"
{TEXT}
\"\"\"
""".strip()

# SEPARATE AGE PROMPTS  (predict age only)

SEPARATE_AGE_BASELINE = """
You are an expert in linguistic profiling.

Predict the age group of the author of the following text.

Allowed labels:
0-25
26-35
36-65
66-

Rules:
- Choose exactly one label.
- If uncertain, choose the most likely option.
- Output only in the required format, nothing else.

Output format:
Age: <0-25 | 26-35 | 36-65 | 66->

Text:
\"\"\"
{TEXT}
\"\"\"
""".strip()

SEPARATE_AGE_STYLE = """
You are a linguistic analyst specialized in author profiling.

Determine the author's age group based ONLY on writing style cues.

Focus on:
- vocabulary and word choice
- use of slang or formal language
- punctuation and capitalization
- sentence length and structure
- tone and emotional expression
- degree of formality

Allowed labels:
0-25
26-35
36-65
66-

Rules:
- Choose exactly one label.
- If uncertain, choose the most likely option.
- Do not explain your reasoning.
- Output only in the required format, nothing else.

Output format:
Age: <0-25 | 26-35 | 36-65 | 66->

Text:
\"\"\"
{TEXT}
\"\"\"
""".strip()

SEPARATE_AGE_DEMOGRAPHIC = """
You are a linguistic analyst specialized in demographic author profiling.

Predict the age group of the author based on linguistic patterns.

Age markers to consider:
- 0-25: casual language, internet slang, abbreviations,
  short sentences, informal tone, minimal punctuation
- 26-35: mix of formal and informal, complete sentences, references to work or adult life
- 36-65: more formal vocabulary, longer sentences, traditional expressions, less slang
- 66-: very formal language, old-fashioned expressions,
  avoids slang, longer and more complex sentences

Allowed labels:
0-25
26-35
36-65
66-

Rules:
- Choose exactly one label.
- If uncertain, choose the most likely option.
- Do not explain your reasoning.
- Output only in the required format, nothing else.

Output format:
Age: <0-25 | 26-35 | 36-65 | 66->

Text:
\"\"\"
{TEXT}
\"\"\"
""".strip()

# few-shot joint prompt
# {EXAMPLES} is filled at runtime from fewshot_examples.csv
# {TEXT} is the test instance

JOINT_FEWSHOT = """
You are a linguistic analyst specialized in stylometric author profiling.

Predict the gender and age group of an author based on their writing style.

Look for stylometric signals:
- Vocabulary richness and word choice (formal vs. casual, slang, abbreviations)
- Sentence length and syntactic complexity
- Punctuation and capitalisation habits
- Hedging, assertiveness, emotional expression
- Generational markers (internet slang, cultural references, spelling conventions)

Allowed labels:
Gender: M  F
Age:    0-25  26-35  36-65  66-

Output format (exactly two lines, nothing else):
Gender: <M or F>
Age: <0-25 | 26-35 | 36-65 | 66->

Examples:
{EXAMPLES}
Now predict for this text:
Text:
\"\"\"
{TEXT}
\"\"\"
""".strip()

# experiment registry

JOINT_PROMPTS = {
    "joint_baseline": JOINT_BASELINE,
    "joint_style": JOINT_STYLE,
    "joint_demographic": JOINT_DEMOGRAPHIC,
}

SEPARATE_GENDER_PROMPTS = {
    "separate_baseline": SEPARATE_GENDER_BASELINE,
    "separate_style": SEPARATE_GENDER_STYLE,
    "separate_demographic": SEPARATE_GENDER_DEMOGRAPHIC,
}

SEPARATE_AGE_PROMPTS = {
    "separate_baseline": SEPARATE_AGE_BASELINE,
    "separate_style": SEPARATE_AGE_STYLE,
    "separate_demographic": SEPARATE_AGE_DEMOGRAPHIC,
}

JOINT_FEWSHOT_V2 = """
You are a linguistic analyst specialized in stylometric author profiling.
Predict the gender and age group of an author based on their writing style.

Look for stylometric signals:
- Vocabulary richness and word choice (formal vs. casual, slang, abbreviations)
- Sentence length and syntactic complexity
- Punctuation and capitalisation habits
- Hedging, assertiveness, emotional expression
- Generational markers (internet slang, cultural references, spelling conventions)

Allowed labels:
Gender: M  F
Age:    0-25  26-35  36-65  66-

Class distribution note: approximately 47% of authors are aged 36-65, 37% are
26-35, 10% are 66-, and only 5% are 0-25. Predict 0-25 only when you see
unmistakable youth signals (heavy slang, abbreviations, emojis, no punctuation).

Output format (exactly two lines, nothing else):
Gender: <M or F>
Age: <0-25 | 26-35 | 36-65 | 66->

Examples:
{EXAMPLES}

Now predict for this text:
Text:
\"\"\"
{TEXT}
\"\"\"
""".strip()

# FEW-SHOT V3  (chain-of-thought + separate signals + targeted 66- recovery)
#
# Key changes from v2:
#   - Model outputs brief stylometric observations BEFORE labels (CoT)
#   - Examples include observation traces showing HOW to reason
#   - Explicit 66- marker checklist with concrete linguistic examples
#   - Contrastive note distinguishing 36-65 from 66- for female authors
#   - max_tokens must be raised to ~80 in the runner script

JOINT_FEWSHOT_V3 = """
You are a forensic linguist conducting academic research on authorship attribution.
Predict the GENDER and AGE GROUP of the author based exclusively on writing style.

Step 1 — note 2-3 stylometric signals in the text.
Step 2 — predict gender and age.

Gender signals:
- M: direct assertions, minimal hedging, blunt or aggressive phrasing, internet spelling (ur, dont, wont)
- F: hedging ("I think", "maybe", "perhaps"), emotional qualifiers, politeness markers, elaborate phrasing

Age signals:
- 0-25: emoji, hashtags, heavy slang/abbreviations (lol, omg, tbh, ur), very short sentences, no punctuation
- 26-35: mix of formal/informal, complete sentences, adult-life references (work, politics), occasional slang
- 36-65: formal vocabulary, complete complex sentences, policy/political framing, minimal slang
- 66-: OLD-FASHIONED formal register — "I am getting cross" not "I'm angry", subjunctive mood,
  rhetorical questions in series, no abbreviations whatsoever, hypercorrect punctuation,
  explicit historical references, very long elaborate paragraphs, British formal expressions

CRITICAL: do not confuse 36-65 with 66-. If the text uses any informal contractions (I'm, don't)
or modern internet vocabulary, it is NOT 66-. Predict 66- only for unmistakably archaic formal register.

Allowed labels:  Gender: M  F  |  Age: 0-25  26-35  36-65  66-

Output format (exactly three lines, nothing else):
Signals: <2-3 brief stylometric observations>
Gender: <M or F>
Age: <0-25 | 26-35 | 36-65 | 66->

Examples:

Text:
\"\"\"
I hope your wife marries another woman. \U0001f60e #gaypolygamy
\"\"\"
Signals: emoji present, hashtag, single very short sentence, zero punctuation beyond period, provocative casual tone
Gender: F
Age: 0-25

Text:
\"\"\"
I love Europe and it's the reason I detest the EU. Left wing fascism at it's worse. Merkel has caused this ridiculous scenario and then a week later closes her borders. Germany IS the EU and act like dictators to the rest of Europe. It's time Europe got out of this tyranny
\"\"\"
Signals: direct aggressive assertions without hedging, modern political vocabulary (EU, Merkel), sustained argument with complete sentences, adult political framing
Gender: M
Age: 26-35

Text:
\"\"\"
Soon, oh so soon germany will understand israel. By the time merkel gets it, it will be too late.
\"\"\"
Signals: hedged temporal framing ("soon, oh so soon"), qualified pessimistic prediction, moderate formality, no slang
Gender: F
Age: 36-65

Text:
\"\"\"
Strange how for months we have had news reports and film of people, women, children, whole families drowned at sea while looking for refuge and one simple photo of a little boy lying on the beach, dead starts all this. Mmmmm.
\"\"\"
Signals: reflective "Mmmmm" as deliberate pause marker, complex embedded relative clauses, formal detached tone, no contractions or slang
Gender: M
Age: 66-

Text:
\"\"\"
I'm sorry your score wasn't high enough to get a job better than 11 B and that your intimidated by a woman being next to you in your foxhole doing the exact same job as you, possibly even better than you because she's not held back by her antiquated ideals.
\"\"\"
Signals: formal ironic apology framing, elaborate conditional syntax, educated vocabulary ("antiquated"), assertive feminist argument — modern register despite formality
Gender: F
Age: 36-65

Text:
\"\"\"
Hmmmm... I am getting cross... I have just read that Syria launched air attacks against IS, or whatever they call themselves. My question is WHY haven't Syria formed an ARMY...??? This is an important issue because lots of their young men are running away... WHY are they not in an army...? WHY should they look for a 'better life' in Europe (and the UK) when we had to fight for our way of life, with death and injury affecting our families...??? Now with US and Russia involved, it seems to me that WW3 is approaching... I SUGGEST that Europe (and the UK) accept women, with their children, as refugees.
\"\"\"
Signals: "I am getting cross" (archaic British idiom), series of rhetorical WHY questions with ellipsis, uppercase emphasis throughout, explicit WW2/WW3 historical framing, long elaborate paragraph — unmistakably archaic register
Gender: F
Age: 66-

Now predict for this text:
Text:
\"\"\"
{TEXT}
\"\"\"
""".strip()
