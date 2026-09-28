# Long text KV eviction experiment

## Question

Can a standard text-cache policy be applied after Q-ViK visual eviction without
discarding Q-ViK's selected visual evidence?

The implemented order is:

1. prefill the multimodal prompt;
2. score and evict visual entries with Q-ViK;
3. mark every surviving visual entry as protected;
4. apply StreamingLLM or H2O-style selection only to text entries during
   greedy decoding.

H2O-style selection uses cumulative per-head decode attention. It intentionally
does not request the full prefill attention matrix, whose memory is quadratic
in prompt length.

## Benchmark priority

1. **MM-NIAH text-needle retrieval, counting, and reasoning.** This gives a
   controlled sweep over text context length and needle depth, which is the
   cleanest direct test of whether text eviction loses evidence in the middle.
   Use the public validation split for ablations.
2. **MMLongBench text-RAG / text-DocQA and Many-Shot ICL.** These tasks expose
   explicit long text together with visual input and have 8K/16K and longer
   configurations. Use the text-heavy task configs before the vision-heavy
   aggregate.
3. **MMLongBench-Doc with OCR text.** The documents average roughly 21K text
   tokens and 47.5 pages. Filter by evidence source (plain text, table, chart)
   and report cross-page results separately.
4. **MileBench diagnostic NIAH.** Keep this as a secondary benchmark because
   much of MileBench's length comes from multiple images rather than a long
   tokenized text prompt.

Always log the tokenizer-derived text length after chat templating. Dataset
names alone do not guarantee that text, rather than image patches, dominates
the final model sequence.

The downloaded MM-NIAH validation annotations have the following
`context_length_text` distribution:

| Task | n | median | p90 | max |
| --- | ---: | ---: | ---: | ---: |
| retrieval-text | 519 | 9,939 | 44,876 | 56,876 |
| counting-text | 517 | 10,764 | 44,412 | 55,797 |
| reasoning-text | 520 | 9,856 | 45,161 | 58,607 |

This makes the full split a substantially better test of text eviction than
the short VQA tasks. After applying the actual Vicuna chat template and
expanding every image placeholder to 576 tokens, only 32 retrieval, 44
counting, and 45 reasoning examples fit within 2,048 tokens. Use those rows
only as an implementation sanity check; the main result must use a
long-context model over the same complete examples.

## Ablation matrix

Use identical examples, generation settings, and visual/text budgets:

| Run | Visual cache | Text cache |
| --- | --- | --- |
| Full | full | full |
| Q-ViK | Q-ViK | full |
| StreamingLLM | full | sinks + recent |
| H2O | full | heavy + recent |
| Sequential-S | Q-ViK | sinks + recent |
| Sequential-H | Q-ViK | heavy + recent |

Sweep text budgets at 80%, 50%, and 20%. For H2O, split each total text budget
equally between heavy and recent entries (`h2o_recent_ratio=0.5`), matching the
common 10% heavy + 10% recent setup at a 20% total budget. Also include fixed
cache sizes so examples of different lengths are compared at the same memory
budget.

Report task accuracy, peak allocated GPU memory, prefill latency, decode
latency/token, and the fields written to `*_keep_ratio_stats.json`.

The standalone runner writes both a detailed `predictions.jsonl` and the five
fields expected by the upstream MM-NIAH scorer under `official_outputs/`.
For text retrieval/reasoning it applies the benchmark's VQA-style normalized
word-containment score. For counting it parses the requested JSON list and
uses per-item soft accuracy.

## Model limit

The local LLaVA-1.5-7B config declares 4,096 rotary positions, but its tokenizer
and current builder expose a 2,048-token evaluation limit. It is suitable only
for the <=2K sanity slice. Use `keep_ratio_basis=image` so a long text prompt
does not consume the visual-token budget.
StreamingLLM and H2O are cache policies; they do not by themselves extend that
native window. Use the OneVision/long-context model for the main 8K+ sweep.

The current second stage runs after full prompt prefill, so it reduces
post-prefill KV memory and decode cost, not peak prefill attention cost. A claim
about prefill scalability requires a separate chunked-prefill implementation
and should be reported separately.
