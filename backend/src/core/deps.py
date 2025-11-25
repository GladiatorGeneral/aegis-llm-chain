from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict:
    '''Get current authenticated user from JWT token'''
    # Mock implementation - replace with real JWT validation
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Not authenticated'
        )
    
    # Return mock user for now
    return {
        'id': '1',
        'username': 'demo_user',
        'email': 'demo@example.com'
    }

async def get_current_active_user(
    current_user: Dict = Depends(get_current_user)
) -> Dict:
    '''Get current active user'''
    if not current_user:
        raise HTTPException(status_code=400, detail='Inactive user')
    return current_user
