from typing import Optional, List

from sqlalchemy.orm import Session

from app.core.security import get_password_hash, verify_password
from app.models.user import User
from app.schemas.user import UserCreate, UserUpdate

class UserService:
    """Service for user operations."""
    
    @staticmethod
    def get_by_id(db: Session, user_id: int) -> Optional[User]:
        """Get a user by ID."""
        return db.query(User).filter(User.id == user_id).first()
    
    @staticmethod
    def get_by_email(db: Session, email: str) -> Optional[User]:
        """Get a user by email."""
        return db.query(User).filter(User.email == email).first()
    
    @staticmethod
    def get_by_username(db: Session, username: str) -> Optional[User]:
        """Get a user by username."""
        return db.query(User).filter(User.username == username).first()
    
    @staticmethod
    def get_all(db: Session, skip: int = 0, limit: int = 100) -> List[User]:
        """Get all users."""
        return db.query(User).offset(skip).limit(limit).all()
    
    @staticmethod
    def create(db: Session, user_in: UserCreate) -> User:
        """Create a new user."""
        db_user = User(
            email=user_in.email,
            username=user_in.username,
            hashed_password=get_password_hash(user_in.password),
            age=user_in.age,
            is_active=user_in.is_active,
            is_superuser=user_in.is_superuser,
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user
    
    @staticmethod
    def update(db: Session, db_user: User, user_in: UserUpdate) -> User:
        """Update a user."""
        update_data = user_in.dict(exclude_unset=True)
        if update_data.get("password"):
            hashed_password = get_password_hash(update_data["password"])
            del update_data["password"]
            update_data["hashed_password"] = hashed_password
        
        for field, value in update_data.items():
            setattr(db_user, field, value)
        
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user
    
    @staticmethod
    def delete(db: Session, db_user: User) -> User:
        """Delete a user."""
        db.delete(db_user)
        db.commit()
        return db_user
    
    @staticmethod
    def authenticate(db: Session, username: str, password: str) -> Optional[User]:
        """Authenticate a user."""
        user = UserService.get_by_username(db, username)
        if not user:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user 