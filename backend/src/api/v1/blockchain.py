from fastapi import APIRouter

router = APIRouter()

@router.get('/blockchain/health')
async def blockchain_health():
    return {'status': 'ok', 'module': 'blockchain'}
