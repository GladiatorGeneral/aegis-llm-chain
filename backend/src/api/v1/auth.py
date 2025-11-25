from fastapi import APIRouter

router = APIRouter()

@router.get('/auth/health')
async def auth_health():
    return {'status': 'ok', 'module': 'auth'}
