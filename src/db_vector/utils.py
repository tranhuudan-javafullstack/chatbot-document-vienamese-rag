from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from src.config.app_config import get_settings

settings = get_settings()

from itertools import cycle

api_keys = ["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", "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",
            "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", "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",
            "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",
            "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", "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"]
api_key_cycle = cycle(api_keys)
import time


def generate_embeddings(text, max_retries=3, retry_delay=2):
    for _ in range(max_retries):
        try:
            current_api_key = next(api_key_cycle)  # Lấy API key tiếp theo
            embeddings = HuggingFaceInferenceAPIEmbeddings(
                api_key=current_api_key,
                model_name=settings.MODEL_EMBEDDING_NAME
            )
            return embeddings.embed_query(text)
        except Exception as e:
            print(f"API key {current_api_key} failed. Trying the next one...")
            time.sleep(retry_delay)  # Tạm dừng trước khi thử lại

    # Nếu tất cả các API key đều thất bại
    raise Exception("All API keys failed. Please check your keys or try again later.")


def get_recursive_token_chunk(chunk_size=256):
    MARKDOWN_SEPARATORS = [
        "\n#{1,6} ",
        "```\n",
        "\n\\*\\*\\*+\n",
        "\n---+\n",
        "\n___+\n",
        "\n\n",
        "\n",
        " ",
        "",
    ]
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(settings.MODEL_EMBEDDING_NAME),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )
    return text_splitter
