import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def get_models_dir() -> Path:
    """Get the models directory path, creating if needed."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR


def check_models_installed() -> dict:
    """Check which models are installed."""
    embeddings_dir = MODELS_DIR / "embeddings"
    llm_dir = MODELS_DIR / "llm" / "phi-3-mini"
    
    return {
        "embeddings": embeddings_dir.exists() and list(embeddings_dir.glob("*")),
        "llm": llm_dir.exists() and list(llm_dir.glob("*"))
    }
