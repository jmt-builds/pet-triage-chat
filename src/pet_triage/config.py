"""Runtime settings, read from environment variables named like the fields (``MODEL``)."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Settings for the triage service.

    Attributes:
        urgent_care_phone_number: Number given to owners for urgent cases. The default is
            from Ofcom's range reserved for drama, so it can never ring a real line.
        client_id: The signed-in client, whose animals the call can be about. In
            production it comes from the user's session; the demo fixes it.
        model: The Ollama model used for extraction: a tag, or a Hugging Face GGUF repo.
        thinking: Whether the model thinks before it answers. Its answer is still held to
            the schema; the thinking only comes first. With ``gemma4:e4b``, on an earlier
            set of evaluation cases, it cut per-message under-triage from 27 to 8, at about
            1 second per message instead of 0.4.
        temperature: The model's sampling temperature. 0 makes its readings repeatable;
            some thinking models, like Qwen, loop at 0 and need their recommended value.
        ollama_base_url: Where the Ollama server listens. Compose points it at its own
            Ollama service.
        timeout_seconds: How long one model call may take before it fails, so a hung
            model can't leave the owner waiting forever.
    """

    urgent_care_phone_number: str = "01632 960000"
    client_id: str = "client-001"
    model: str = "gemma4:e4b"
    thinking: bool = True
    temperature: float = 0.0
    ollama_base_url: str = "http://localhost:11434"
    timeout_seconds: float = 60.0
