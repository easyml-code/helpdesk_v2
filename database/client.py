from supabase import create_client, Client
from fastapi import FastAPI, HTTPException
from database.utils import get_new_tokens, pg_escape
from logs.log import logger, log_query, set_trace_id
from metrics.prometheus import track_db_query, track_error
from config import settings
from typing import Tuple, List, Dict, Any, Optional
from fastapi.concurrency import run_in_threadpool
from psycopg import OperationalError
import traceback
import psycopg
import json
import jwt
import time


# Create a Supabase client factory (sync SDK)
def make_supabase_client() -> Client:
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)


async def get_access_token(email: str, password: str) -> Tuple[str, str]:
    """Authenticate user and return tokens"""
    trace_id = set_trace_id()
    supabase = make_supabase_client()

    try:
        auth_res = await run_in_threadpool(
            supabase.auth.sign_in_with_password,
            {"email": email, "password": password}
        )
    except Exception as exc:
        logger.error(f"auth_failed - email={email}, trace_id={trace_id}", exc_info=True)
        track_error("auth_failure", "supabase")
        raise HTTPException(status_code=500, detail="Authentication service error")

    session = getattr(auth_res, "session", None)
    if not session:
        logger.info(f"auth_invalid_credentials - email={email}, trace_id={trace_id}")
        track_error("invalid_credentials", "auth")
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = getattr(session, "access_token", None)
    refresh_token = getattr(session, "refresh_token", None)

    if not access_token or not refresh_token:
        logger.error(f"auth_missing_tokens - email={email}, trace_id={trace_id}")
        track_error("missing_tokens", "auth")
        raise HTTPException(status_code=500, detail="Failed to obtain tokens")

    logger.info(
        f"auth_success - email={email}, uid={getattr(session.user, 'id', 'unknown')}, "
        f"trace_id={trace_id}"
    )
    return access_token, refresh_token


async def run_query(
    query: str,
    access_token: str,
    refresh_token: Optional[str] = None,
    *,
    retry_on_expire: bool = True
) -> List[Dict[str, Any]]:
    """Execute database query with metrics and logging"""
    
    if not access_token:
        logger.error("run_query_missing_token")
        raise HTTPException(status_code=400, detail="Missing access token")

    # Determine query type for metrics
    query_type = query.strip().split()[0].upper()
    start_time = time.time()

    def _connect_and_exec(access_token: str) -> List[Dict[str, Any]]:
        conn = None
        cur = None
        exec_start = time.time()
        
        try:
            # Decode JWT
            decoded = jwt.decode(
                access_token,
                settings.JWT_SECRET,
                algorithms=["HS256"],
                audience="authenticated",
                options={"verify_aud": True}
            )
            
        except jwt.InvalidTokenError as e:
            logger.error(f"invalid_jwt - error={e}")
            track_error("invalid_jwt", "database")
            raise HTTPException(status_code=401, detail="Invalid access token")
        
        try:
            # Connect to database
            conn = psycopg.connect(
                host=settings.SUPABASE_HOST,
                port=5432,
                dbname=settings.SUPABASE_DB,
                user=settings.SUPABASE_USER,
                password=settings.POSTGRES_PASSWORD,
                sslmode="require",
                connect_timeout=5
            )
        
            cur = conn.cursor()

            # Set role and JWT claims
            cur.execute("SET ROLE authenticated;")
            jwt_claims_json = json.dumps(decoded)
            decoded_claims = jwt_claims_json.replace("'", "''")
            cur.execute(f"SET LOCAL request.jwt.claims = '{decoded_claims}';")

            # Execute query
            cur.execute(query)
            
            # Commit for INSERT/UPDATE/DELETE
            if cur.description is None:
                conn.commit()
                execution_time = (time.time() - exec_start) * 1000
                
                # Log and track metrics
                log_query(query, execution_time, 0)
                track_db_query(query_type, execution_time / 1000, 0, success=True)
                
                return []

            # SELECT or INSERT ... RETURNING rows
            rows = cur.fetchall()
            conn.commit()

            # Convert rows to list[dict]
            desc = [col.name for col in cur.description] if cur.description else []
            result: List[Dict[str, Any]] = []
            for row in rows:
                row_dict = {desc[i]: row[i] for i in range(len(desc))}
                result.append(row_dict)

            execution_time = (time.time() - exec_start) * 1000
            
            # Log and track metrics
            log_query(query, execution_time, len(result))
            track_db_query(query_type, execution_time / 1000, len(result), success=True)
            
            return result
            
        except OperationalError as oe:
            execution_time = (time.time() - exec_start) * 1000
            logger.error(f"db_operational_error - error={oe}, time={execution_time}ms")
            track_db_query(query_type, execution_time / 1000, 0, success=False)
            track_error("operational_error", "database")
            raise
            
        except Exception as exc:
            execution_time = (time.time() - exec_start) * 1000
            logger.error(f"db_query_error - error={exc}, time={execution_time}ms", exc_info=True)
            track_db_query(query_type, execution_time / 1000, 0, success=False)
            track_error("query_error", "database")
            raise HTTPException(status_code=500, detail="Error executing query")

        finally:
            if cur is not None:
                cur.close()
            if conn is not None:
                conn.close()

    try:
        result = await run_in_threadpool(_connect_and_exec, access_token)
        total_time = (time.time() - start_time) * 1000
        logger.info(f"query_success - rows={len(result)}, total_time={total_time:.2f}ms")
        return result

    except OperationalError as oe:
        logger.warning(f"db_auth_error - attempting_token_refresh - error={oe}")
        
        if not refresh_token or not retry_on_expire:
            logger.error("token_refresh_unavailable - aborting")
            track_error("auth_failed_no_refresh", "database")
            raise HTTPException(status_code=401, detail="Database authentication failed")

        # Attempt token refresh
        logger.info("token_refresh_attempt")
        try:
            new_access, new_refresh = await get_new_tokens(make_supabase_client(), refresh_token)
            if not new_access:
                logger.error("token_refresh_failed - no_new_token")
                track_error("token_refresh_failed", "database")
                raise HTTPException(status_code=401, detail="Failed to refresh token")

            # Retry with new token
            try:
                result = await run_in_threadpool(_connect_and_exec, new_access)
                total_time = (time.time() - start_time) * 1000
                logger.info(
                    f"query_success_after_refresh - rows={len(result)}, "
                    f"total_time={total_time:.2f}ms"
                )
                return result
            except OperationalError as oe2:
                logger.error(f"db_auth_failed_after_refresh - error={oe2}", exc_info=True)
                track_error("auth_failed_after_refresh", "database")
                raise HTTPException(status_code=401, detail="Database auth failed after token refresh")
                
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(f"token_refresh_exception - error={exc}", exc_info=True)
            track_error("token_refresh_exception", "database")
            raise HTTPException(status_code=500, detail="Failed to refresh access token")

    except Exception as exc:
        total_time = (time.time() - start_time) * 1000
        logger.error(f"query_unexpected_error - error={exc}, time={total_time:.2f}ms", exc_info=True)
        track_error("unexpected_error", "database")
        raise HTTPException(status_code=500, detail="Unexpected error executing query")