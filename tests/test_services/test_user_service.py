import pytest
from sqlalchemy.orm import Session

from app.services.user import UserService
from app.schemas.user import UserCreate, UserUpdate
from app.models.user import User

def test_create_user(db: Session):
    """Test creating a user."""
    user_in = UserCreate(
        email="test@example.com",
        username="testcreate",
        password="password123",
    )
    user = UserService.create(db, user_in=user_in)
    
    assert user.email == user_in.email
    assert user.username == user_in.username
    assert hasattr(user, "hashed_password")
    assert user.hashed_password != user_in.password

def test_get_by_id(db: Session, normal_user: dict):
    """Test getting a user by ID."""
    user = UserService.get_by_id(db, user_id=normal_user["id"])
    
    assert user is not None
    assert user.id == normal_user["id"]
    assert user.email == normal_user["email"]
    assert user.username == normal_user["username"]

def test_get_by_email(db: Session, normal_user: dict):
    """Test getting a user by email."""
    user = UserService.get_by_email(db, email=normal_user["email"])
    
    assert user is not None
    assert user.id == normal_user["id"]
    assert user.email == normal_user["email"]
    assert user.username == normal_user["username"]

def test_get_by_username(db: Session, normal_user: dict):
    """Test getting a user by username."""
    user = UserService.get_by_username(db, username=normal_user["username"])
    
    assert user is not None
    assert user.id == normal_user["id"]
    assert user.email == normal_user["email"]
    assert user.username == normal_user["username"]

def test_get_all(db: Session, normal_user: dict, superuser: dict):
    """Test getting all users."""
    users = UserService.get_all(db)
    
    assert len(users) >= 2
    assert any(u.id == normal_user["id"] for u in users)
    assert any(u.id == superuser["id"] for u in users)

def test_update_user(db: Session, normal_user: dict):
    """Test updating a user."""
    user = UserService.get_by_id(db, user_id=normal_user["id"])
    
    user_update = UserUpdate(
        email="updated_test@example.com",
    )
    updated_user = UserService.update(db, db_user=user, user_in=user_update)
    
    assert updated_user.id == user.id
    assert updated_user.email == user_update.email
    assert updated_user.username == user.username

def test_update_user_password(db: Session, normal_user: dict):
    """Test updating a user's password."""
    user = UserService.get_by_id(db, user_id=normal_user["id"])
    original_hashed_password = user.hashed_password
    
    new_password = "newpassword123"
    user_update = UserUpdate(
        password=new_password,
    )
    updated_user = UserService.update(db, db_user=user, user_in=user_update)
    
    assert updated_user.id == user.id
    assert updated_user.hashed_password != original_hashed_password
    
    # Test authentication with new password
    authenticated_user = UserService.authenticate(
        db, username=normal_user["username"], password=new_password
    )
    assert authenticated_user is not None
    assert authenticated_user.id == user.id

def test_delete_user(db: Session):
    """Test deleting a user."""
    # Create a user to delete
    user_in = UserCreate(
        email="delete@example.com",
        username="deleteuser",
        password="password123",
    )
    user = UserService.create(db, user_in=user_in)
    
    # Delete the user
    deleted_user = UserService.delete(db, db_user=user)
    
    assert deleted_user.id == user.id
    
    # Verify user is deleted
    user_check = UserService.get_by_id(db, user_id=user.id)
    assert user_check is None

def test_authenticate_success(db: Session, normal_user: dict):
    """Test successful authentication."""
    user = UserService.authenticate(
        db, username=normal_user["username"], password=normal_user["password"]
    )
    
    assert user is not None
    assert user.id == normal_user["id"]
    assert user.email == normal_user["email"]
    assert user.username == normal_user["username"]

def test_authenticate_fail_wrong_password(db: Session, normal_user: dict):
    """Test authentication with wrong password."""
    user = UserService.authenticate(
        db, username=normal_user["username"], password="wrongpassword"
    )
    
    assert user is None

def test_authenticate_fail_nonexistent_user(db: Session):
    """Test authentication with nonexistent user."""
    user = UserService.authenticate(
        db, username="nonexistentuser", password="password123"
    )
    
    assert user is None 