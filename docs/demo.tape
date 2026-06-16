# KairoRM demo recording — produces docs/demo.gif
#
#   brew install vhs        # one time
#   vhs docs/demo.tape      # run from the repo root, with a working key in .env
#
# Timing is adaptive (Wait+Screen), so it works regardless of how long the scan
# or the LLM takes — no fixed Sleep to tune.

Output docs/demo.gif

Set Shell "bash"
Set FontSize 16
Set Width 1280
Set Height 760
Set Padding 20
Set Theme "Catppuccin Mocha"

# Hidden setup: use Groq's high-throughput model so the parallel agents never hit
# a per-minute burst limit, and force HF offline (model is cached) so no hub
# warning leaks. `clear` wipes these lines before recording resumes.
Hide
Type "export KAIRO_GROQ_MODEL=llama-3.1-8b-instant HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
Enter
Type "clear"
Enter
Show

# Kick off the scan (kairo loads GROQ_API_KEY from ./.env)
Type ".venv/bin/kairo map https://github.com/karpathy/micrograd"
Enter

# Wait until the interactive console has rendered (its prompt placeholder appears)
Wait+Screen@180s /Ask about this codebase/
Sleep 2s

# Ask a question; the answer streams in token by token
Escape
Type "How does backpropagation work in this codebase?"
Enter
Sleep 1s
Wait+Screen@90s /code chunk/
Sleep 4s

# Quit the console
Ctrl+C
Sleep 1s
