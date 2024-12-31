from minio import Minio
from minio.error import S3Error
import io
from datetime import datetime
from src.config.app_config import get_settings
import json

settings = get_settings()
bucket_name = settings.BUCKET_NAME

minio_client = Minio(
    f"{settings.MINIO_HOST}:{settings.MINIO_PORT}",
    access_key=settings.MINIO_ACCESS_KEY,
    secret_key=settings.MINIO_SECRET_ACCESS_KEY,
    secure=False  # Set to True if you are using HTTPS,
    , region=settings.REGION_NAME,
)


def set_public_read_policy(bucket_name, object_name):
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": ["*"]},
                "Action": ["s3:GetObject"],
                "Resource": [f"arn:aws:s3:::{bucket_name}/{object_name}"]
            }
        ]
    }
    policy_str = json.dumps(policy)
    minio_client.set_bucket_policy(bucket_name, policy_str)


def upload_to_minio(doc_name, file_type, local_file):
    file_counter = 0
    file_name = local_file.split('/')[-1]
    file_name_without_ext, file_extension = file_name.rsplit('.', 1)

    while True:
        file_name_with_counter = f'{file_name_without_ext}_{file_counter}.{file_extension}'
        minio_file_with_counter = f'{doc_name}/{file_type}/{get_current_time()}/{file_name_with_counter}'

        if not check_file_exists(minio_file_with_counter):
            break
        file_counter += 1

    url = upload_file_to_minio(local_file, minio_file_with_counter)
    return url, minio_file_with_counter


def create_bucket_if_not_exist():
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name, object_lock=False)


def get_current_time():
    return datetime.now().strftime("%Y-%m-%d")


def check_file_exists(minio_file_with_counter):
    try:
        minio_client.stat_object(bucket_name, minio_file_with_counter)
        return True
    except S3Error as e:
        if e.code == 'NoSuchKey':
            return False
        raise


def upload_file_to_minio(local_file, minio_file_with_counter):
    try:
        minio_client.fput_object(bucket_name, minio_file_with_counter, local_file)
        print(f"Upload thành công! Tên file trên MinIO: {minio_file_with_counter}")
        return f'http://minio:9000/{bucket_name}/{minio_file_with_counter}'
    except FileNotFoundError:
        print("File không tồn tại")
    except S3Error as e:
        print(f"Lỗi MinIO: {e}")
    except Exception as e:
        print(f"Lỗi: {e}")
    return None


def upload_file_to_minio_with_counter(key_name, user_name, folder_name, file_bytes, file_name, file_type):
    file_counter = 0
    file_name_without_ext, file_extension = file_name.rsplit('.', 1)

    while True:
        file_name_with_counter = f'{file_name_without_ext}_{file_counter}.{file_extension}'
        minio_file_path = f'{user_name}/{key_name}/{folder_name}/{get_current_time()}/{file_type}/{file_name_with_counter}'
        if not check_file_exists(minio_file_path):
            break
        file_counter += 1

    try:
        file_data = io.BytesIO(file_bytes)
        file_size = len(file_bytes)
        minio_client.put_object(bucket_name=bucket_name, object_name=minio_file_path, data=file_data, length=file_size)
        print(f"Upload thành công! Tên file trên MinIO: {minio_file_path}")
        url = f'{settings.SERVER_IP}:{settings.MINIO_PORT}/{bucket_name}/{minio_file_path}'
        set_public_read_policy(bucket_name, minio_file_path)
        return url, minio_file_path
    except S3Error as e:
        print(f"Lỗi MinIO: {e}")
    except Exception as e:
        print(f"Lỗi: {e}")
    return None, None


def upload_avatar_to_minio_with_counter(key_name, user_name, folder_name, file_bytes, file_name):
    file_counter = 0
    file_name_without_ext, file_extension = file_name.rsplit('.', 1)

    while True:
        file_name_with_counter = f'{file_name_without_ext}_{file_counter}.{file_extension}'
        minio_file_path = f'{user_name}/{key_name}/{folder_name}/{file_name_with_counter}'
        if not check_file_exists(minio_file_path):
            break
        file_counter += 1

    try:
        file_data = io.BytesIO(file_bytes)
        file_size = len(file_bytes)
        minio_client.put_object(bucket_name=bucket_name, object_name=minio_file_path, data=file_data, length=file_size)
        print(f"Upload thành công! Tên file trên MinIO: {minio_file_path}")
        url = f'http://localhost:9000/{bucket_name}/{minio_file_path}'
        set_public_read_policy(bucket_name, minio_file_path)
        return url, minio_file_path
    except S3Error as e:
        print(f"Lỗi MinIO: {e}")
    except Exception as e:
        print(f"Lỗi: {e}")
    return None, None


def upload_user_avatar_to_minio(user_name, file_bytes, file_name):
    return upload_avatar_to_minio_with_counter("avatar", user_name, f'avatar_user', file_bytes, file_name)


def upload_bot_avatar_to_minio(user_name, folder_name, file_bytes, file_name):
    return upload_avatar_to_minio_with_counter("avatar", user_name, f'avatar_bot/{folder_name}', file_bytes, file_name)


def upload_file_knowledge_to_minio(key_name, user_name, knowledge_name, file_bytes, file_name, file_type):
    return upload_file_to_minio_with_counter(key_name, user_name, knowledge_name, file_bytes, file_name, file_type)


def delete_from_minio(minio_file_with_counter):
    try:
        minio_client.remove_object(bucket_name, minio_file_with_counter)
        print(f"File '{minio_file_with_counter}' đã được xoá khỏi bucket '{bucket_name}'.")
        return True
    except S3Error as e:
        print(f"Lỗi: {e}")
        return False


def delete_folder_from_minio(folder_path):
    try:
        objects_to_delete = minio_client.list_objects(bucket_name, prefix=folder_path, recursive=True)
        for obj in objects_to_delete:
            minio_client.remove_object(bucket_name, obj.object_name)
        print(f"Thư mục '{folder_path}' đã được xóa khỏi bucket '{bucket_name}'.")
        return True
    except S3Error as e:
        print(f"Lỗi khi xóa thư mục: {e}")
        return False
    except Exception as e:
        print(f"Lỗi không xác định khi xóa thư mục: {e}")
        return False


def read_file_as_bytes(file_path):
    with open(file_path, 'rb') as file:
        file_bytes = file.read()
    return file_bytes
