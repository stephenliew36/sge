# SGE: Sequential Guided Execution

`sge_pipline/sge.py` builds the logistic feature vector and routes at P > 0.5. `sge_pipline/pipeline.py` sends simple queries to the LLM, and on the reasoning path has the LLM write a plan that the LRM refines. `sge_pipline/evaluate.py` scores accuracy (set `SGE_JUDGE=1` to use GPT-4.1). Set `OPENROUTER_API_KEY` or `OPENAI_API_KEY` before a model run.
