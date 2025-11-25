from fastapi import APIRouter

router = APIRouter()

@router.get('/cognitive/health')
async def cognitive_health():
    return {'status': 'ok', 'module': 'cognitive'}
