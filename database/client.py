from supabase import create_client, Client
from fastapi import FastAPI, HTTPException
from database.utils import get_new_tokens, pg_escape

from logs.log import logger
from config import settings
from typing import Tuple, List, Dict, Any, Optional
from fastapi.concurrency import run_in_threadpool
from psycopg import OperationalError
import traceback
import psycopg
import json
import jwt
import os


# Create a Supabase client factory (sync SDK)
def make_supabase_client() -> Client:
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

async def get_access_token(email: str, password: str) -> Tuple[str, str]:
 
    supabase = make_supabase_client()

    try:
        auth_res = await run_in_threadpool(
            supabase.auth.sign_in_with_password,
            {"email": email, "password": password}
        )
    except Exception as exc:
        logger.exception("Supabase sign-in failed for email=%s", email)
        raise HTTPException(status_code=500, detail="Authentication service error")

    session = getattr(auth_res, "session", None)
    if not session:
        logger.info("Sign-in failed or no session returned for email=%s", email)
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = getattr(session, "access_token", None)
    refresh_token = getattr(session, "refresh_token", None)

    if not access_token or not refresh_token:
        logger.error("Auth session missing tokens for email=%s: %s", email, auth_res)
        raise HTTPException(status_code=500, detail="Failed to obtain tokens")

    logger.info("User signed in: email=%s (uid=%s)", email, getattr(session.user, "id", "unknown"))
    return access_token, refresh_token

async def run_query(
    query: str,
    access_token: str,
    refresh_token: Optional[str] = None,
    *,
    retry_on_expire: bool = True
) -> List[Dict[str, Any]]:
    
    if not access_token:
        logger.error("run_query missing access_token")
        raise HTTPException(status_code=400, detail="Missing access token")

    def _connect_and_exec(access_token: str) -> List[Dict[str, Any]]:
        conn = None
        cur = None
        try:
            decoded = jwt.decode(
                access_token,
                settings.JWT_SECRET,
                algorithms=["HS256"],
                audience="authenticated",
                options={"verify_aud": True}
            )
            
        except jwt.InvalidTokenError as e:
            logger.error("Invalid access token: %s", e)
            raise HTTPException(status_code=401, detail="Invalid access token")
        
        try:
            conn = psycopg.connect(
                host=settings.SUPABASE_HOST,
                port=5432,
                dbname=settings.SUPABASE_DB,
                user=settings.SUPABASE_USER,
                password=settings.POSTGRES_PASSWORD,
                sslmode="require",
                connect_timeout=10
            )
        
            cur = conn.cursor()

            cur.execute("SET ROLE authenticated;")
            jwt_claims_json = json.dumps(decoded)

            decoded_claims = jwt_claims_json.replace("'", "''")
            cur.execute(f"SET LOCAL request.jwt.claims = '{decoded_claims}';")

            # Execute query
            cur.execute(query)
            
            # Commit for INSERT/UPDATE/DELETE
            if cur.description is None:
                conn.commit()
                return []

            # SELECT or INSERT ... RETURNING rows
            rows = cur.fetchall()
            conn.commit()

            # convert rows -> list[dict] using description
            desc = [col.name for col in cur.description] if cur.description else []
            result: List[Dict[str, Any]] = []
            for row in rows:
                row_dict = {desc[i]: row[i] for i in range(len(desc))}
                result.append(row_dict)

            return result
        except OperationalError as oe:
            logger.error("OperationalError during DB operation: %s", oe)
            raise
        except Exception as exc:
            logger.exception("Error executing query: %s\n%s", exc, traceback.format_exc())
            raise HTTPException(status_code=500, detail="Error executing query")

        finally:
            if cur is not None:
                cur.close()
            if conn is not None:
                conn.close()

    try:
        result = await run_in_threadpool(_connect_and_exec, access_token)
        logger.info("run_query success (initial token) rows=%d", len(result))
        return result

    except OperationalError as oe:
        logger.warning("DB OperationalError: %s", oe)
        if not refresh_token or not retry_on_expire:
            logger.exception("Auth failed and no refresh token; aborting")
            raise HTTPException(status_code=401, detail="Database authentication failed")

        # attempt refresh
        logger.info("Attempting to refresh access token with refresh_token")
        try:
            new_access, new_refresh = await get_new_tokens(make_supabase_client(), refresh_token)
            if not new_access:
                logger.error("get_new_tokens did not return new access token")
                raise HTTPException(status_code=401, detail="Failed to refresh token")

            # retry with new token
            try:
                result = await run_in_threadpool(_connect_and_exec, new_access)
                logger.info("run_query success after refresh; rows=%d", len(result))
                return result
            except OperationalError as oe2:
                logger.exception("DB auth still failing after token refresh: %s", oe2)
                raise HTTPException(status_code=401, detail="Database auth failed after token refresh")
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Token refresh failed: %s\n%s", exc, traceback.format_exc())
            raise HTTPException(status_code=500, detail="Failed to refresh access token")

    except Exception as exc:
        logger.exception("Unexpected error in run_query: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail="Unexpected error executing query")