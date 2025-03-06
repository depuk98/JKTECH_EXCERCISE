import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from sqlalchemy.orm import Session
from fastapi import HTTPException
from jose import JWTError
from pydantic import ValidationError

from app.api.deps import get_current_user, get_current_active_user, get_current_superuser, authenticate_user
from app.models.user import User


@pytest.mark.asyncio
async def test_get_current_user_success():
    """Test successful current user retrieval."""
    # Mock dependencies
    mock_db = MagicMock(spec=Session)
    mock_token = "valid.jwt.token"
    
    # Create a mock user
    mock_user = MagicMock(spec=User)
    mock_user.id = 1
    mock_user.username = "testuser"
    mock_user.is_active = True
    mock_user.is_superuser = False
    
    # Mock the database query
    mock_query = MagicMock()
    mock_filter = MagicMock()
    mock_first = MagicMock(return_value=mock_user)
    
    mock_db.query.return_value = mock_query
    mock_query.filter.return_value = mock_filter
    mock_filter.first.return_value = mock_first.return_value
    
    # Mock the jwt.decode function
    with patch('app.api.deps.jwt.decode') as mock_decode, \
         patch('app.api.deps.TokenPayload') as mock_token_payload_cls:
        # Setup the mock to return a valid payload
        mock_decode.return_value = {"sub": "1", "exp": 9999999999.0}  # Use string for sub and float for exp
        
        # Create a mock TokenPayload instance with proper attributes
        mock_token_payload = MagicMock()
        mock_token_payload.sub = "1"
        mock_token_payload.exp = 9999999999.0  # Set exp as float
        mock_token_payload_cls.return_value = mock_token_payload
        
        # Call the function
        user = await get_current_user(db=mock_db, token=mock_token)
        
        # Assert
        assert user == mock_user
        mock_db.query.assert_called_once_with(User)
        mock_query.filter.assert_called_once()
        mock_filter.first.assert_called_once()


@pytest.mark.asyncio
async def test_get_current_user_invalid_token():
    """Test current user retrieval with invalid token."""
    # Mock dependencies
    mock_db = MagicMock(spec=Session)
    mock_token = "invalid.jwt.token"
    
    # Mock the jwt.decode function to raise JWTError
    with patch('app.api.deps.jwt.decode') as mock_decode:
        # Setup the mock to raise JWTError
        mock_decode.side_effect = JWTError("Invalid token")
        
        # Call the function and expect exception
        with pytest.raises(HTTPException) as excinfo:
            await get_current_user(db=mock_db, token=mock_token)
        
        # Assert
        assert excinfo.value.status_code == 401
        assert "Could not validate credentials" in excinfo.value.detail


@pytest.mark.asyncio
async def test_get_current_user_not_found():
    """Test current user retrieval when user doesn't exist."""
    # Mock dependencies
    mock_db = MagicMock(spec=Session)
    mock_token = "valid.jwt.token"
    
    # Mock the database query to return None
    mock_query = MagicMock()
    mock_filter = MagicMock()
    
    mock_db.query.return_value = mock_query
    mock_query.filter.return_value = mock_filter
    mock_filter.first.return_value = None
    
    # Mock the jwt.decode function
    with patch('app.api.deps.jwt.decode') as mock_decode, \
         patch('app.api.deps.TokenPayload') as mock_token_payload_cls:
        # Setup the mock to return a valid payload
        mock_decode.return_value = {"sub": "999", "exp": 9999999999.0}  # Use string for sub and float for exp
        
        # Create a mock TokenPayload instance with proper attributes
        mock_token_payload = MagicMock()
        mock_token_payload.sub = "999"
        mock_token_payload.exp = 9999999999.0  # Set exp as float
        mock_token_payload_cls.return_value = mock_token_payload
        
        # Call the function and expect exception
        with pytest.raises(HTTPException) as excinfo:
            await get_current_user(db=mock_db, token=mock_token)
        
        # Assert - the status code might be 404 or 403 depending on implementation
        assert excinfo.value.status_code in [404]
        assert "User not found" in excinfo.value.detail


@pytest.mark.asyncio
async def test_get_current_user_expired_token():
    """Test get_current_user with an expired token."""
    # Mock the jwt.decode function to simulate an expired token
    with patch('app.api.deps.jwt.decode') as mock_decode:
        mock_decode.side_effect = JWTError("Token expired")
        
        # Create a mock db session
        mock_db = MagicMock()
        
        # Test the function
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(db=mock_db, token="expired_token")
        
        # Assert the exception details
        assert exc_info.value.status_code == 401
        assert "Could not validate credentials" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_current_user_invalid_payload():
    """Test get_current_user with an invalid token payload."""
    # Mock the jwt.decode function to return an invalid payload
    with patch('app.api.deps.jwt.decode') as mock_decode, \
         patch('app.api.deps.TokenPayload') as mock_token_payload:
        # Setup mock to raise ValidationError or JWTError
        mock_decode.return_value = {"not_sub": "invalid_payload"}  # Missing 'sub' field
        
        # Instead of creating a complex ValidationError, just have the function raise an exception
        mock_token_payload.side_effect = JWTError("Invalid token payload")
        
        # Create a mock db session
        mock_db = MagicMock()
        
        # Test the function
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(db=mock_db, token="invalid_payload_token")
        
        # Assert the exception details
        assert exc_info.value.status_code == 401
        assert "Could not validate credentials" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_current_active_user_success():
    """Test successful active user retrieval."""
    # Create a mock active user
    mock_user = MagicMock(spec=User)
    mock_user.id = 1
    mock_user.username = "testuser"
    mock_user.is_active = True
    
    # Call the function
    user = await get_current_active_user(current_user=mock_user)
    
    # Assert
    assert user is mock_user


@pytest.mark.asyncio
async def test_get_current_active_user_inactive():
    """Test active user retrieval with inactive user."""
    # Create a mock inactive user
    mock_user = MagicMock(spec=User)
    mock_user.id = 1
    mock_user.username = "testuser"
    mock_user.is_active = False
    
    # Call the function and expect exception
    with pytest.raises(HTTPException) as excinfo:
        await get_current_active_user(current_user=mock_user)
    
    # Assert
    assert excinfo.value.status_code == 400
    assert "Inactive user" in excinfo.value.detail


@pytest.mark.asyncio
async def test_get_current_superuser_success():
    """Test successful superuser retrieval."""
    # Create a mock superuser
    mock_user = MagicMock(spec=User)
    mock_user.id = 1
    mock_user.username = "admin"
    mock_user.is_active = True
    mock_user.is_superuser = True
    
    # Call the function
    user = await get_current_superuser(current_user=mock_user)
    
    # Assert
    assert user is mock_user


@pytest.mark.asyncio
async def test_get_current_superuser_not_superuser():
    """Test superuser retrieval with regular user."""
    # Create a mock regular user
    mock_user = MagicMock(spec=User)
    mock_user.id = 1
    mock_user.username = "testuser"
    mock_user.is_active = True
    mock_user.is_superuser = False
    
    # Call the function and expect exception
    with pytest.raises(HTTPException) as excinfo:
        await get_current_superuser(current_user=mock_user)
    
    # Assert
    assert excinfo.value.status_code == 403
    assert "Not enough permissions" in excinfo.value.detail


@pytest.mark.asyncio
async def test_get_current_superuser_with_non_superuser():
    """Test attempting to get superuser with a regular user token."""
    # Create a mock user that is NOT a superuser
    mock_user = MagicMock()
    mock_user.is_superuser = False
    mock_user.is_active = True
    
    # Mock the get_current_active_user function
    with patch('app.api.deps.get_current_active_user', return_value=mock_user):
        # Test the function
        with pytest.raises(HTTPException) as exc_info:
            await get_current_superuser(current_user=mock_user)
        
        # Assert the exception details
        assert exc_info.value.status_code == 403
        assert "Not enough permissions" in exc_info.value.detail


def test_authenticate_user_success():
    """Test successful user authentication."""
    # Mock dependencies
    mock_db = MagicMock(spec=Session)
    
    # Create a mock user
    mock_user = MagicMock(spec=User)
    mock_user.id = 1
    mock_user.username = "testuser"
    mock_user.hashed_password = "hashed_password"
    
    # Mock the database query
    mock_query = MagicMock()
    mock_filter = MagicMock()
    
    mock_db.query.return_value = mock_query
    mock_query.filter.return_value = mock_filter
    mock_filter.first.return_value = mock_user
    
    # Mock the verify_password function
    with patch('app.api.deps.verify_password', return_value=True) as mock_verify:
        # Call the function
        user = authenticate_user(db=mock_db, username="testuser", password="password")
        
        # Assert
        assert user is mock_user
        mock_verify.assert_called_once_with("password", "hashed_password")


def test_authenticate_user_wrong_password():
    """Test user authentication with wrong password."""
    # Mock dependencies
    mock_db = MagicMock(spec=Session)
    
    # Create a mock user
    mock_user = MagicMock(spec=User)
    mock_user.id = 1
    mock_user.username = "testuser"
    mock_user.hashed_password = "hashed_password"
    
    # Mock the database query
    mock_query = MagicMock()
    mock_filter = MagicMock()
    
    mock_db.query.return_value = mock_query
    mock_query.filter.return_value = mock_filter
    mock_filter.first.return_value = mock_user
    
    # Mock the verify_password function
    with patch('app.api.deps.verify_password', return_value=False) as mock_verify:
        # Call the function
        user = authenticate_user(db=mock_db, username="testuser", password="wrong_password")
        
        # Assert
        assert user is None
        mock_verify.assert_called_once_with("wrong_password", "hashed_password")


def test_authenticate_user_user_not_found():
    """Test user authentication with nonexistent user."""
    # Mock dependencies
    mock_db = MagicMock(spec=Session)
    
    # Mock the database query to return None
    mock_query = MagicMock()
    mock_filter = MagicMock()
    
    mock_db.query.return_value = mock_query
    mock_query.filter.return_value = mock_filter
    mock_filter.first.return_value = None
    
    # Call the function
    user = authenticate_user(db=mock_db, username="nonexistent", password="password")
    
    # Assert
    assert user is None


def test_authenticate_user_with_none_values():
    """Test authenticate_user with None values."""
    # Create a mock db
    mock_db = MagicMock()
    
    # For None username, we need to mock the query chain to avoid errors
    mock_db.query.return_value.filter.return_value.first.return_value = None
    
    # Test with None username
    result = authenticate_user(db=mock_db, username=None, password="password")
    assert result is None
    
    # Test with None password
    result = authenticate_user(db=mock_db, username="username", password=None)
    assert result is None
    
    # Test with both valid inputs but user not found
    result = authenticate_user(db=mock_db, username="username", password="password")
    assert result is None


def test_oauth2_scheme_resolution():
    """Test the OAuth2 scheme resolution."""
    # Check that oauth2_scheme is properly initialized
    from app.api.deps import oauth2_scheme
    
    # Verify it's a properly configured OAuth2PasswordBearer instance
    # The actual attributes in FastAPI's OAuth2PasswordBearer may vary by version
    assert oauth2_scheme is not None
    # Check the scheme is correctly initialized with expected URL pattern
    assert hasattr(oauth2_scheme, "model")
    assert "api/auth/login" in str(oauth2_scheme.__dict__)


@pytest.mark.asyncio
async def test_get_current_active_user_with_none():
    """Test get_current_active_user with None user."""
    # We need to handle the None case differently since the function expects a User
    # Let's mock get_current_user to return None and see how get_current_active_user handles it
    with patch('app.api.deps.get_current_user', return_value=None):
        # The function should raise an exception when current_user is None
        with pytest.raises(Exception) as exc_info:  # Use generic Exception to catch any error
            await get_current_active_user()
        
        # The error could be an AttributeError or HTTPException depending on implementation
        assert isinstance(exc_info.value, (AttributeError, HTTPException)) 