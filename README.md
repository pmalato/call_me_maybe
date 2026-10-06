*This project has been created as part of the 42 curriculum by pecoelho.*

# call me maybe

Introduction to function calling in LLMs, using constrained decoding with
**Qwen/Qwen3-0.6B**.

## Description

Large language models understand natural language but do not reliably
produce machine-readable output; a 0.6B model asked for JSON may succeed
only a fraction of the time. The goal of this project is to translate a
natural-language request into a structured **function call** (a function
name plus typed arguments) with **100% valid, schema-compliant JSON**,
without relying on the model to spontaneously write correct JSON.

Given:

```
"What is the sum of 40 and 2?"
```

the program does not answer `42`; it produces:

```json
{
  "prompt": "What is the sum of 40 and 2?",
  "name": "fn_add_numbers",
  "parameters": {"a": 40.0, "b": 2.0}
}
```

Overview of the project:

| File | Role |
| --- | --- |
| `src/__main__.py` | Entry point; runs the pipeline over all prompts. |
| `src/parser.py` | CLI arguments, reading/validating inputs, vocabulary loading, writing the output file. |
| `src/models.py` | Pydantic models (`FunctionValidator`, `PromptValidator`, `OutputValidator`, ...). |
| `src/constrained_decoding.py` | The constrained decoder (function choice and argument generation). |
| `llm_sdk/` | Provided wrapper around the model (`Small_LLM_Model`). |

## Instructions

Requirements: Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
make install      # uv sync (creates the venv and installs dependencies)
make run          # runs the program with the default data/ paths
make debug        # runs the program under pdb
make lint         # flake8 + mypy
make lint-strict  # flake8 + mypy --strict
make clean        # removes caches
```

Direct usage:

```bash
uv run python -m src \
    [--functions_definition <function_definition_file>] \
    [--input <input_file>] \
    [--output <output_file>]
```

Default paths:

- `--functions_definition`: `data/input/functions_definition.json`
- `--input`: `data/input/function_calling_tests.json`
- `--output`: `data/output/function_calling_results.json`

The output directory is created automatically if it does not exist.

## Example usage

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calls.json
```

Input prompts (`function_calling_tests.json`):

```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "Reverse the string 'hello'"}
]
```

Output (`function_calls.json`):

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {"s": "hello"}
  }
]
```

While running, each result is also printed to the terminal, followed by the
total execution time in seconds.

## Algorithm explanation

At every step an LLM produces **logits**: one score per token of the
vocabulary. Constrained decoding intervenes *before* the next token is
chosen: tokens that would break the expected structure are excluded, and
the best remaining token (greedy / argmax) is selected. The generation is
split into three phases.

### 1. Preparation (once)

- The vocabulary file (`get_path_to_vocab_file`) is read, and every token id
  is decoded to its real text, so byte-level markers such as `Ġ` (space) and
  `Ċ` (newline) become actual characters.
- The token ids of each function name are precomputed with `encode`.
- Tokens are classified into sets used later as filters: `digits`, `minus`,
  `dot`, `bool` (`true` / `false`), `str` (tokens that are safe inside a JSON
  string: no `"`, no `\`, no control characters), `str_end` (tokens starting
  with `"`) and `num_end` (tokens starting with `,` or `}`). Special tokens
  such as `<|im_start|>` are ignored.

### 2. Choosing the function

The context is a chat-style prompt: a system message listing the available
functions (name, parameter types, description), the user's request, and the
beginning of the answer `{"name": "`. Then, token by token:

1. The model computes the logits for the current sequence.
2. Only tokens that continue the **prefix of at least one real function
   name** keep their logit; all others are set to `-inf`.
3. The best remaining token is appended.
4. The loop stops when the generated ids are exactly a function name.

The model therefore really *chooses* the function (no heuristics), but it
can only ever produce a name that exists in the definitions.

### 3. Generating the arguments

The JSON skeleton is written **by the program**, not by the model:
`"parameters": {`, each key `"a": `, the opening quote for strings, and the
separators are appended to the context as encoded text. The model only
generates the *values*, following the type declared for each parameter:

| Type | Allowed tokens |
| --- | --- |
| `boolean` | `true` or `false` only (one token). |
| `integer` | digits; `-` only as the first character; a terminator (`,` / `}`) once at least one digit exists. |
| `number` | same as integer, plus a single `.` after a digit, and no terminator right after the dot. |
| `string` | any token without `"`, `\` or control characters, or a token starting with `"`, which closes the string. |

When a terminator token is selected the value is finished (the terminator
itself is discarded since the program writes the separators). Values are
converted to the correct Python type (`str`, `bool`, `int`, `float`) and the
final entry is validated with Pydantic before being written with
`TypeAdapter.dump_json`. Because the file is serialised by Pydantic from
typed values, it is always valid JSON that matches the schema.

## Design decisions

- **The program writes the structure, the model fills the content.**
  Keys, quotes and separators are never left to the model, so structural
  errors are impossible by construction.
- **Token-level filtering with precomputed sets.** The vocabulary is
  classified once, so each step only needs set unions and an `argmax`
  over a small set (`max(allowed, key=...)`), instead of scanning the
  whole vocabulary.
- **Function names as token-id prefixes.** Comparing id sequences is exact
  and cheap, and guarantees the result is always one of the known functions.
- **Hard limits** (`MAX_NAME_TOKENS = 32`, `MAX_VALUE_TOKENS = 64`) prevent
  infinite loops if the model never emits a terminator.
- **Pydantic everywhere** (as required): inputs, function definitions and
  the output entries are validated models; function names must start with
  `fn_`; parameter types are restricted to `string`, `number`, `integer`
  and `boolean`; extra keys in a parameter definition are rejected.
- **Only public SDK methods are used**: `get_logits_from_input_ids`,
  `get_path_to_vocab_file`, `encode` and `decode`.
- **Output order and typing**: parameters are generated in the order they
  appear in the function definition, and numbers declared as `number` are
  returned as floats (`2.0`), as in the subject's example.

## Performance analysis

> Fill in with your own measurements before submitting.

- **Accuracy:** `[X]/[N]` prompts with the correct function and arguments
  (`[X]%`, target: 90%+).
- **Reliability:** 100% of the outputs are valid JSON and schema-compliant,
  since the structure is built by the program and validated with Pydantic.
- **Speed:** `[T]` seconds for `[N]` prompts on `[hardware]` (target: under
  5 minutes). Each generated token costs one forward pass of the model, so
  the time grows with the number of prompts, the number of parameters and the
  length of the values. Precomputing the token sets avoids repeated
  vocabulary scans during generation.
- **Known limitations:**
  - Strings cannot contain `"` or `\` (these tokens are excluded to keep the
    JSON valid without implementing escaping).
  - If one function name is a strict prefix of another, generation stops at
    the shorter name.
  - Accuracy depends on the 0.6B model's understanding of the prompt; the
    constraints guarantee validity, not semantic correctness.

## Challenges faced

> Adapt to your real experience.

- **Byte-level tokens.** The vocabulary file contains markers such as `Ġ`
  and `Ċ` rather than real characters. Decoding every id with `decode`
  solved this and made it possible to classify tokens by their text.
- **Numbers split across tokens.** A number may be produced by several
  tokens, so validity must be checked incrementally (a minus only at the
  start, a single dot, no terminator before a digit).
- **Knowing when a value ends.** Numbers and strings have no fixed length;
  terminator tokens (`,`, `}` or a quote-leading token) are used to detect
  the end.
- **Keeping the model on track.** The prompt ends with the start of the JSON
  answer, so the model continues from the right place.

## Testing strategy

> Adapt to the tests you actually ran.

- Ran the provided examples and checked that the output file exists, is
  valid JSON and contains exactly the keys `prompt`, `name` and `parameters`.
- Verified that each `name` exists in the definitions and that each argument
  matches the declared type (`number`, `integer`, `string`, `boolean`).
- Edge cases: empty strings, large and negative numbers, decimals, special
  characters, ambiguous prompts, functions with several parameters and
  functions with no matching prompt.
- Error handling: missing input files, invalid JSON, wrong schema.
- Static checks: `make lint` (flake8 and mypy).

## Resources

- [Qwen3 model card (Qwen/Qwen3-0.6B)](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [uv documentation](https://docs.astral.sh/uv/)
- [Byte-level BPE tokenization (Hugging Face course)](https://huggingface.co/learn/llm-course/chapter6/5)
- Articles on constrained / structured decoding (logit masking, grammar-based
  decoding).

### How AI was used

> Replace this with an accurate description of your own use of AI.

AI was used for: `[e.g. drafting this README and the docstrings; explaining
the tokenizer and vocabulary format; reviewing ideas for constrained
decoding]`. All AI-generated content was reviewed, tested and is understood
by the author; the decoding algorithm and its implementation were written
and verified by the author.