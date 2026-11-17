# PhD reading group — three lessons

The first meeting introduces the essential concepts without advance reading. The second and third meetings discuss selected research-paper excerpts, connecting their methods to their evidence.

| Lesson | Preparation | Discussion focus |
|---|---|---|
| [01 — Shared foundations](01-foundations.md) | No advance reading | Generative distributions, conditioning, learning from examples and evaluating outputs |
| [02 — How generative models work](02-generative-models.md) | Specified AEVB and DDPM excerpts, listed at the top of the lesson | Hidden variables, training objectives, sampling and what the reported comparisons establish |
| [03 — Preferences and research evidence](03-preferences-and-evidence.md) | Specified DPO and RAG excerpts, listed at the top of the lesson | Preference labels, conditional generation, evaluator choices and retrieval ablations |

For Lessons 2 and 3, the preparation lists give fixed paper versions and exact reading boundaries. They are the complete required reading; the rest of each lesson is supporting explanation and optional extension. Discussion questions name the relevant paper section, equation, algorithm or result and include expandable worked reasoning. The PDFs include those worked responses in full.

Numerical examples labelled illustrative or hypothetical are calculations for discussion, not reported paper results. Repository labs are optional demonstrations, not reproductions of the papers' full experiments.

## Assessment — 30 points

Lesson 1 introduces the foundations and is **ungraded**, with no advance reading or submission. Lessons 2 and 3 each discuss **two papers**. Assessment rewards understanding the readings, explaining the evidence and developing a research question; running a repository experiment is not required.

| Component | Points | What to do |
|---|---|---|
| Two individual reading notes | 6 | Before Lesson 2 and before Lesson 3, submit one 200–300-word note focused on **one of that meeting's two assigned papers**. Each note is worth 3 points. Read the specified excerpts from both papers, even though the note focuses on one. |
| Presentation or discussion leadership | 8 | Take one assessed role in either Lesson 2 or Lesson 3: explain a specific claim and its evidence, or lead a discussion of them. Identify the relevant passage, figure, table or equation. Each student receives a comparable allocated opportunity under the same rubric. |
| Discussion contributions | 6 | Earn up to 3 points in each of Lessons 2 and 3 through questions, reasoned comments and engagement with others' arguments. Short written contributions are an equivalent option. |
| Individual final proposal | 10 | After Lesson 3, submit a two-page proposal testing or extending a specific claim from **any one of the four assigned papers**. Include the question, proposed comparison, measurements and limitations. No implementation is required. |
| **Total** | **30** | |

### Reading notes — 3 points each

Each 200–300-word note should contain all three elements below. Give the paper title and a section, equation, figure or table reference so the claim and evidence can be located.

- **Claim or contribution — 1 point:** explain one specific claim in your own words. “The paper introduces a new model” is too broad; identify what the method or comparison is claimed to achieve.
- **Supporting evidence — 1 point:** identify a relevant derivation, algorithm or reported result from the assigned excerpt, and explain how it supports the claim. Distinguish something the authors demonstrate from something they assume.
- **Justified question — 1 point:** ask one question and explain why its answer matters. For example, “Would the ranking change with a different evaluator?” becomes a useful research question when you identify which reported comparison depends on that evaluator.

Uncertainty is acceptable: name the exact step you do not understand and explain what you can follow up to that point. A general summary without a specific claim and evidence does not satisfy all three criteria.

### Presentation or discussion leadership — 8 points

The same criteria apply whether you present a passage or lead its discussion:

- **Understanding — 3 points:** explain the claim, the necessary concepts and the assumptions or conditions under which it holds.
- **Clear explanation grounded in the paper — 3 points:** walk through a particular equation, algorithm or result, define its terms and connect it to the claim. For discussion leadership, use focused questions and a synthesis that establish these connections.
- **Responses to questions — 2 points:** address challenges with reasons or evidence, distinguish what the paper establishes from your interpretation, and acknowledge unresolved points accurately.

You are not expected to defend every claim in the paper or invent an answer you do not know. Identifying a genuine limitation and explaining what would resolve it demonstrates understanding.

### Discussion contributions — 3 points per meeting

The score reflects the quality of contributions, not how often or how loudly you speak:

- **Relevant reasoning — 1 point:** contribute a focused question or reasoned comment about a claim, assumption or method.
- **Connection to evidence — 1 point:** support the contribution with a specific passage, result or concrete counterexample.
- **Engagement — 1 point:** respond to another participant's argument, refine a question after discussion, or explain how the discussion changed your interpretation.

A short written contribution submitted by the end of the meeting can demonstrate the same criteria, including a response to an argument raised during the session. The assessed presentation or leadership segment is scored under its own rubric; discussion points reflect contributions outside that segment, so the same performance is not counted twice.

### Final proposal — 10 points

Use the two pages to make a small, testable study understandable:

- **Precise research question — 2 points:** state what you want to find out and what outcome would answer it.
- **Connection to a paper's claim — 2 points:** identify the claim and its location, then explain what your study would test or extend.
- **Comparison and measurements — 4 points:** describe the proposed data or examples, the alternatives being compared, what you would keep fixed, and how the measurements would distinguish the alternatives. For example, comparing evaluators should use the same generated responses; otherwise a changed score could reflect changed outputs rather than changed evaluation.
- **Limitations and interpretation — 2 points:** explain what different possible results would mean, one important limitation, and what the study could not establish.

Propose a study; do not invent results. No code, trained model or completed experiment is required. A focused, justified comparison can earn full credit without proposing a new architecture. For each criterion above, full points require a clear and supported answer; incomplete or partly correct reasoning earns partial credit, and an absent answer earns zero.

[Installation for optional computational examples](../docs/setup.md) · [Existing course path](../lessons/README.md)
