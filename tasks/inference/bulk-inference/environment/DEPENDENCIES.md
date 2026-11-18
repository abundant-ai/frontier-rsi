# Baked external assets

The Docker build pins `Qwen/Qwen2.5-1.5B-Instruct` to Hugging Face commit `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. The SHA-256 digests of every selected model and tokenizer file are in `model.sha256`; the build verifies them before the task image is used. The model weights file digest is `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`.

The build installs `@openai/codex@0.159.2` as an agent harness dependency. npm registry integrity values for that exact version are:

- `@openai/codex@0.159.2`: `sha512-SE13C3nZCYoVL569BdegoOl6vwjb7o2sXOo7ivwVzaVoY0cswwi0/6pIE0TyO/C0vIkQh3jslExitET7PBTfIg==`
- Linux x64 native package `@openai/codex@0.159.2-linux-x64`: `sha512-RrCZ1X52wpa1lOsXtCtSyhjOFdQPh7LH5Ccv8HsKmd/2UXbUwxXFqWXFK3JzatquUNGtW/TLox5Y7qVOGkV0/Q==`

The model revision and npm integrity values came from the respective Hugging Face and npm registry metadata. The model file SHA-256 values were computed from downloads at the pinned revision.
