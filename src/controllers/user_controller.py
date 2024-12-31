from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from src.config.app_config import get_settings
from src.db_vector.weaviate_rag_non_tenant import aggregate_for_user
from src.dtos.schema_in.user import UserChangePass, UserUpdate
from src.dtos.schema_out.user import UserBotOut, UserOut, UserKnowledgeOut
from src.models.all_models import User
from src.security import get_current_user
from src.services.user_service import UserService
from src.utils.app_util import get_key_name_minio
from src.utils.minio_util import delete_from_minio, upload_user_avatar_to_minio

settings = get_settings()

user_router = APIRouter()


@user_router.get('/me', summary='Get details of currently logged in user', response_model=UserOut)
async def get_me(user: User = Depends(get_current_user)):
    return user


@user_router.get('/aggregate', summary='Get aggregate of currently logged in user', response_model=UserOut)
async def get_aggregate_me(user: User = Depends(get_current_user)):
    return aggregate_for_user(user.username)


@user_router.get("/bots", summary="Get all bots of user", response_model=UserBotOut)
async def get_bots(user: User = Depends(get_current_user)):
    return await UserService.get_bots(user)


@user_router.get("/knowledges", summary="Get all knowledges of user", response_model=UserKnowledgeOut)
async def get_knowledges(user: User = Depends(get_current_user)):
    return await UserService.get_knowledges(user)


@user_router.put('/update', summary='Update user details', response_model=UserOut)
async def update_user(data: UserUpdate, user: User = Depends(get_current_user)):
    return await UserService.update_user(user.user_id, data)


@user_router.put('/change-pass', summary='Change user password', response_model=UserOut)
async def change_pass(data: UserChangePass, user: User = Depends(get_current_user)):
    return await UserService.change_pass(user.user_id, data)


@user_router.get('/change-avatar-random', summary='Change user password', response_model=UserOut)
async def change_avatar_random(user: User = Depends(get_current_user)):
    return await UserService.change_avatar_random(user.user_id)


@user_router.put('/upload-avatar', summary="Upload user avatar", response_model=UserOut)
async def upload_avatar(user: User = Depends(get_current_user), file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        file_name = file.filename
        user = await User.find_one(User.user_id == user.user_id)
        url, s3_file_path = upload_user_avatar_to_minio(user.username, file_bytes, file_name)
        if url and s3_file_path:
            delete_from_minio(get_key_name_minio(user.avatar))
            user.avatar = url
            await user.save()
            return JSONResponse(content={"url": url, "file_path": s3_file_path}, status_code=200)
        else:
            raise HTTPException(status_code=500, detail="Failed to upload file to S3")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
