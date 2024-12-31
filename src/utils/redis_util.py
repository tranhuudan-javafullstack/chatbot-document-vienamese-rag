import json
import time
from datetime import timedelta, datetime
from typing import Any, Awaitable, List, Dict
from uuid import UUID

import redis

from src.config.app_config import get_settings
from src.dtos.schema_in.query import ConversationItem

settings = get_settings()
redis_client = redis.StrictRedis(host=settings.REDIS_HOST, port=settings.REDIS_PORT,
                                 password=settings.REDIS_PASSWORD, db=0,
                                 decode_responses=True)


def is_allowed(key: str, max_calls: int, time_frame: int) -> bool:
    current_time = int(datetime.now().timestamp())
    redis_key = f"rate_limit:{key}"

    # Sử dụng pipeline để thực hiện các lệnh liên tiếp
    pipeline = redis_client.pipeline()
    pipeline.zremrangebyscore(redis_key, 0, current_time - time_frame)
    pipeline.zcard(redis_key)
    pipeline.zadd(redis_key, {str(current_time): current_time})
    pipeline.expire(redis_key, time_frame)
    results = pipeline.execute()

    call_count = results[1]
    return call_count < max_calls


def update(key: str, time_frame: int):
    current_time = int(datetime.now().timestamp())
    redis_key = f"rate_limit:{key}"

    pipeline = redis_client.pipeline()
    pipeline.zadd(redis_key, {str(current_time): current_time})
    pipeline.expire(redis_key, time_frame)
    pipeline.execute()


def reset(key: str):
    redis_key = f"rate_limit:{key}"
    redis_client.delete(redis_key)


def set_user_token_in_redis(user_id: str, token_type: str, token: str, expires_delta: timedelta):
    key = f"user:{user_id}:{token_type}"
    redis_client.set(key, token)
    redis_client.expire(key, int(expires_delta.total_seconds()))


def get_user_token_from_redis(user_id: str, token_type: str) -> tuple[Awaitable, Awaitable | Any] | tuple[None, None]:
    key = f"user:{user_id}:{token_type}"
    token = redis_client.get(key)
    if token:
        remaining_time = redis_client.ttl(key)
        return token, remaining_time
    return None, None


def delete_user_token_from_redis(user_id: str, token_type: str):
    key = f"user:{user_id}:{token_type}"
    redis_client.delete(key)


def set_user_history_chat(user_id: str, chat_id: str, message: str, role: str, query_id: UUID, max_length: int = 50):
    key = f"user:{user_id}:chat:{chat_id}"
    chat_entry = {
        "query_id": str(query_id),
        "message": message,
        "role": role,
        "timestamp": int(time.time())
    }
    with redis_client.pipeline() as pipe:
        pipe.rpush(key, json.dumps(chat_entry))
        pipe.ltrim(key, -max_length, -1)
        pipe.execute()


def get_user_history_chat(user_id: str, chat_id: str, limit: int = 10) -> List[Dict]:
    key = f"user:{user_id}:chat:{chat_id}"
    chat_history = redis_client.lrange(key, -limit, -1)
    chat_history = [json.loads(entry) for entry in chat_history]
    return chat_history


def convert_chat_history_to_items(user_id: str, chat_id: str, limit: int = 10) -> List[ConversationItem]:
    chat_history = get_user_history_chat(user_id, chat_id, limit)
    conversation_items = [
        ConversationItem(message=entry['message'], role=entry['role'])
        for entry in chat_history
    ]
    return conversation_items


def update_user_history_chat(user_id: str, chat_id: str, query_id: UUID, new_message: str, role: str) -> bool:
    key = f"user:{user_id}:chat:{chat_id}"
    chat_history = redis_client.lrange(key, 0, -1)

    for index, entry in enumerate(chat_history):
        chat_entry = json.loads(entry)
        if chat_entry['query_id'] == str(query_id) and chat_entry['role'] == role:
            chat_entry['message'] = new_message
            redis_client.lset(key, index, json.dumps(chat_entry))
            return True
    return False


def delete_user_history_chat_by_query_id(user_id: str, chat_id: str, query_id: UUID):
    key = f"user:{user_id}:chat:{chat_id}"
    chat_history = redis_client.lrange(key, 0, -1)
    for entry in chat_history:
        chat_entry = json.loads(entry)
        if chat_entry['query_id'] == str(query_id):
            redis_client.lrem(key, 1, entry)


def delete_user_history_chat(user_id: str, chat_id: str):
    key = f"user:{user_id}:chat:{chat_id}"
    redis_client.delete(key)


if __name__ == '__main__':
    print(convert_chat_history_to_items("e3e556eb-273b-4ed4-9021-a0ba90c3c42c", "3bad30e8-ac66-4103-b5df-2c9c00eac45a"))
