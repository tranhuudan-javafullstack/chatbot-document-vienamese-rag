import random
import re
import secrets
import time

import tiktoken

from src.config.app_config import get_settings

# Constants
AVATAR_NAMES = [
    "Samantha", "Daisy", "Simba", "Jasmine", "Boo", "Bailey", "Toby", "Sugar", "Spooky", "Cuddles",
    "Buddy", "Oscar", "Boots", "Sheba", "Harley", "Buster", "Scooter", "Mimi", "Abby", "Tinkerbell"
]
AVATAR_NAMES_BOT = [
    "Simba", "Simba", "Missy", "Princess", "Rascal", "Lucky", "Sadie", "Luna", "Harley", "Midnight",
    "Loki", "Bubba", "Max", "Oliver", "Bailey", "Chloe"
]


# Function definitions
def strip_non_letters(s: str) -> str:
    """Replace non-alphanumeric characters in string with underscores."""
    return re.sub(r"[^a-zA-Z0-9]", "_", s)


def count_token(string: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(string))


def unique_string(byte: int = 8) -> str:
    """Generate a unique string of specified byte length."""
    return secrets.token_urlsafe(byte)


def generate_unique_code(email: str) -> str:
    """Generate a unique verification code based on email and settings."""
    current_time = int(time.time() * 1000)
    secret_key = get_settings().SECRET_KEY
    combined_string = f"{email}{secret_key}{current_time}"
    combined_hash = hashlib.md5(combined_string.encode()).hexdigest()
    hash_int = int(combined_hash[:8], 16)
    code = (current_time + hash_int) % 1000000
    return f"{code:06d}"


def generate_random_password(min_length: int = 8, max_length: int = 12) -> str:
    """Generate a random password with specified length range."""
    special_characters = string.punctuation
    digits = string.digits
    uppercase_letters = string.ascii_uppercase
    lowercase_letters = string.ascii_lowercase

    password = [
        random.choice(special_characters),
        random.choice(digits),
        random.choice(uppercase_letters),
        random.choice(lowercase_letters)
    ]

    remaining_length = random.randint(min_length, max_length) - len(password)
    all_characters = special_characters + digits + uppercase_letters + lowercase_letters
    password += random.choices(all_characters, k=remaining_length)
    random.shuffle(password)

    return ''.join(password)


def get_random_avatar() -> str:
    """Generate a random avatar URL."""
    avatar_random = random.choice(AVATAR_NAMES)
    avatar_name = f"https://api.dicebear.com/9.x/adventurer/svg?seed={avatar_random}"
    return avatar_name


def get_random_avatar_bot() -> str:
    """Generate a random avatar URL."""
    avatar_random_bot = random.choice(AVATAR_NAMES_BOT)
    avatar_name = f"https://api.dicebear.com/9.x/identicon/svg?seed={avatar_random_bot}"
    return avatar_name


import string


def valid_password(value):
    if not any(char.isdigit() for char in value):
        raise ValueError('Password must contain at least one digit')
    if not any(char.isupper() for char in value):
        raise ValueError('Password must contain at least one uppercase letter')
    if not any(char in r'!@#$%^&*()-_=+{}[]|;:"<>,.?/' for char in value):
        raise ValueError('Password must contain at least one special character')
    return value


def valid_file_or_folder_name(name, allow_hidden=True):
    # Kiểm tra độ dài tên
    if len(name) == 0 or len(name) > 25:
        raise ValueError('Name must be between 1 and 25 characters long')

    # Kiểm tra ký tự không hợp lệ
    invalid_chars = set(r'\/:*?"<>| ')
    if any(char in invalid_chars for char in name):
        raise ValueError(f'Name contains invalid characters: {invalid_chars}')

    # Kiểm tra tên không được chỉ chứa dấu chấm hoặc khoảng trắng
    if all(char in '. ' for char in name):
        raise ValueError('Name cannot consist solely of dots or spaces')

    # Kiểm tra tên không kết thúc bằng dấu chấm hoặc khoảng trắng
    if name[-1] in '. ':
        raise ValueError('Name cannot end with a dot or space')

    if not allow_hidden and name.startswith('.'):
        raise ValueError('Hidden names are not allowed')

    return name


settings = get_settings()


def get_key_name_s3(value: str):
    return value.replace("https://d6ew9gb5lrjk9.cloudfront.net", "")


import hashlib


def generate_username(email):
    local_part, domain = email.split('@')
    domain_hash = hashlib.md5(domain.encode()).hexdigest()[:5]
    username = f"User_{local_part}_{domain_hash}"
    return username


def generate_key_knowledge(name):
    domain_hash = hashlib.md5(str(name).encode()).hexdigest()[:5]
    username = f"Knowledge_{domain_hash}"
    return username


def get_key_name_minio(value: str):
    return value.replace(f"{settings.SERVER_IP}:{settings.MINIO_PORT}/{settings.BUCKET_NAME}/", "")
