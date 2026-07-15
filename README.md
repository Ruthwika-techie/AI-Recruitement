# TechVest Multi-Agent Recruitment Crew

A multi-agent AI system that screens candidates for a **Junior AI Engineer** role using LangGraph. Five specialist agents — Analyst, Scorer, Verifier, Decider, and Scheduler — work in a coordinated pipeline with injection defence, fairness checks, a human approval gate, and a 5-layer evaluation suite.

---

## Project Structure

```
ai-recruitement/
├── app.py               # Streamlit UI (5 tabs)
├── agents.py            # Agent nodes + routing functions
├── crew_graph.py        # LangGraph graph wiring + step budget
├── crew_state.py        # Shared TypedDict state
├── data.py              # JD, candidates, rubric, Pydantic models
├── tools.py             # LangChain tools (parse, score, schedule)
├── guardrails.py        # Injection defence, fairness check, audit log
├── main.py              # CLI runner with human gate prompt
├── graph.py             # Single-agent graph (baseline)
├── state.py             # AgentState for single-agent graph
├── eval_dataset.json    # 10-task evaluation dataset
├── eval_trace.py        # Trace invariants + trajectory judge
├── eval_output.py       # DeepEval faithfulness / relevancy
├── eval_redteam.py      # Red-team probes + Giskard scan
├── eval_governance.py   # Human gate assertions
├── eval_runner.py       # Master eval runner
├── audit/               # Auto-generated run audit logs
└── .streamlit/
    └── config.toml
```

---

## Setup

### 1. Install dependencies

```bash
pip install streamlit langgraph langchain-openai pydantic deepeval python-dotenv python-docx
```

### 2. Configure environment

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://openrouter.ai/api/v1
MODEL_NAME=openai/gpt-4o-mini
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

### 4. Run the CLI pipeline

```bash
python main.py                   # full run with human gate prompt
python main.py --auto-approve    # skip gate (CI / demo mode)
python main.py --fairness-check  # run name-swap fairness test first
```

### 5. Run the evaluation suite

```bash
python eval_runner.py             # all layers
python eval_runner.py --layer 2   # trace only
python eval_runner.py --layer 4   # red-team only
python eval_runner.py --fast      # skip LLM judge + fairness
python eval_runner.py --save      # save scorecard JSON
```
