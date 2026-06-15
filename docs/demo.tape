# KairoRM demo recording — produces docs/demo.gif
#
#   brew install vhs        # one time
#   vhs docs/demo.tape      # run from the repo root, with a working key in .env
#
# Tune the Sleep after `Enter` to match how long your scan actually takes
# (depends on repo size + LLM speed). Use a small repo for a snappy GIF.

Output docs/demo.gif

Set Shell "bash"
Set FontSize 16
Set Width 1280
Set Height 760
Set Padding 20
Set Theme "Catppuccin Mocha"

# Kick off the scan (kairo loads GROQ_API_KEY from ./.env)
Type ".venv/bin/kairo map https://github.com/karpathy/micrograd"
Enter
Sleep 45s

# We're in the interactive console now — ask a question
Escape
Type "How does backpropagation work in this codebase?"
Enter
Sleep 12s

# Quit the console
Ctrl+C
Sleep 2s
