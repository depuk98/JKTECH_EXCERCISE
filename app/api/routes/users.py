from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_active_user, get_current_superuser, get_db
from app.api.deps import get_current_active_user_async, get_current_superuser_async
from app.db.session import get_async_db
from app.models.user import User
from app.schemas.user import User as UserSchema, UserCreate, UserUpdate
from app.services.user import UserService
from app.utils.model_conversion import sqlalchemy_to_pydantic, convert_user_to_dict

router = APIRouter()

@router.get("/me/", response_model=UserSchema)
async def read_user_me(
    current_user: User = Depends(get_current_active_user_async),
) -> Any:
    """
    Get current user.
    """
    # Convert the User model to a dictionary compatible with the Pydantic schema
    user_dict = convert_user_to_dict(current_user)
    return user_dict

@router.get("/me", response_model=UserSchema)
async def read_user_me_no_slash(
    current_user: User = Depends(get_current_active_user_async),
) -> Any:
    """
    Get current user (endpoint without trailing slash).
    """
    # Convert the User model to a dictionary compatible with the Pydantic schema
    user_dict = convert_user_to_dict(current_user)
    return user_dict

@router.put("/me/", response_model=UserSchema)
async def update_user_me(
    *,
    db: AsyncSession = Depends(get_async_db),
    user_in: UserUpdate,
    current_user: User = Depends(get_current_active_user_async),
) -> Any:
    """
    Update current user.
    """
    user = await UserService.update_async(db, db_user=current_user, user_in=user_in)
    # Convert the User model to a dictionary compatible with the Pydantic schema
    user_dict = convert_user_to_dict(user)
    return user_dict

@router.put("/me", response_model=UserSchema)
async def update_user_me_no_slash(
    *,
    db: AsyncSession = Depends(get_async_db),
    user_in: UserUpdate,
    current_user: User = Depends(get_current_active_user_async),
) -> Any:
    """
    Update current user (endpoint without trailing slash).
    """
    user = await UserService.update_async(db, db_user=current_user, user_in=user_in)
    # Convert the User model to a dictionary compatible with the Pydantic schema
    user_dict = convert_user_to_dict(user)
    return user_dict

@router.get("/", response_model=List[UserSchema])
async def read_users(
    db: AsyncSession = Depends(get_async_db),
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_superuser_async),
) -> Any:
    """
    Retrieve users. Only for superusers.
    """
    users = await UserService.get_all_async(db, skip=skip, limit=limit)
    # Convert each User model to a dictionary compatible with the Pydantic schema
    user_dicts = [convert_user_to_dict(user) for user in users]
    return user_dicts

@router.get("/{user_id}", response_model=UserSchema)
async def read_user_by_id(
    user_id: int,
    current_user: User = Depends(get_current_active_user_async),
    db: AsyncSession = Depends(get_async_db),
) -> Any:
    """
    Get a specific user by id.
    """
    user = await UserService.get_by_id_async(db, user_id=user_id)
    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found",
        )
    if user.id != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=403,
            detail="Not enough permissions",
        )
    # Convert the User model to a dictionary compatible with the Pydantic schema
    user_dict = convert_user_to_dict(user)
    return user_dict

@router.post("/", response_model=UserSchema)
async def create_user(
    *,
    db: AsyncSession = Depends(get_async_db),
    user_in: UserCreate,
    current_user: User = Depends(get_current_superuser_async),
) -> Any:
    """
    Create new user. Only for superusers.
    """
    user = await UserService.get_by_email_async(db, email=user_in.email)
    if user:
        raise HTTPException(
            status_code=400,
            detail="A user with this email already exists.",
        )
    user = await UserService.get_by_username_async(db, username=user_in.username)
    if user:
        raise HTTPException(
            status_code=400,
            detail="A user with this username already exists.",
        )
    user = await UserService.create_async(db, user_in=user_in)
    # Convert the User model to a dictionary compatible with the Pydantic schema
    user_dict = convert_user_to_dict(user)
    return user_dict

@router.put("/{user_id}", response_model=UserSchema)
async def update_user(
    *,
    db: AsyncSession = Depends(get_async_db),
    user_id: int,
    user_in: UserUpdate,
    current_user: User = Depends(get_current_superuser_async),
) -> Any:
    """
    Update a user. Only for superusers.
    """
    user = await UserService.get_by_id_async(db, user_id=user_id)
    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found",
        )
    user = await UserService.update_async(db, db_user=user, user_in=user_in)
    # Convert the User model to a dictionary compatible with the Pydantic schema
    user_dict = convert_user_to_dict(user)
    return user_dict

@router.delete("/{user_id}", response_model=UserSchema)
async def delete_user(
    *,
    db: AsyncSession = Depends(get_async_db),
    user_id: int,
    current_user: User = Depends(get_current_superuser_async),
) -> Any:
    """
    Delete a user. Only for superusers.
    """
    user = await UserService.get_by_id_async(db, user_id=user_id)
    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found",
        )
    user = await UserService.delete_async(db, db_user=user)
    # Convert the User model to a dictionary compatible with the Pydantic schema
    user_dict = convert_user_to_dict(user)
    return user_dict 