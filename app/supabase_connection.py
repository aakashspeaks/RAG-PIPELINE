from __future__ import annotations

import os
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()


def get_supabase_client() -> Client:
	"""Create and return a Supabase client from environment variables."""
	supabase_url = os.getenv("SUPABSE_DATABASE_URL")
	supabase_key = os.getenv("SUPABASE_API_KEY")

	if not supabase_url:
		raise ValueError("SUPABASE_URL is missing. Add it to your .env file.")
	if not supabase_key:
		raise ValueError("SUPABASE_KEY is missing. Add it to your .env file.")

	return create_client(supabase_url, supabase_key)


def test_supabase_connection() -> bool:
	"""Run a lightweight connectivity check against Supabase REST endpoint."""
	supabase_url = os.getenv("SUPABSE_DATABASE_URL")
	supabase_key = os.getenv("SUPABASE_API_KEY")

	if not supabase_url or not supabase_key:
		raise ValueError("SUPABASE_DATABASE_URL or SUPABASE_API_KEY is missing.")

	rest_url = f"{supabase_url.rstrip('/')}/rest/v1/"
	request = Request(
		rest_url,
		headers={
			"apikey": supabase_key,
			"Authorization": f"Bearer {supabase_key}",
		},
	)

	with urlopen(request, timeout=10) as response:
		return 200 <= response.status < 300


def test_postgres_uri_connection() -> bool:
	"""Test direct Postgres connection using SUPABASE_DATABASE_URL."""
	db_uri = os.getenv("SUPABSE_DATABASE_URL")
	if not db_uri:
		raise ValueError("SUPABSE_DATABASE_URL is missing. Add it to your .env file.")

	# Lazy import so REST mode still works even if psycopg is not installed.
	try:
		import psycopg
	except ImportError as exc:
		raise ImportError("psycopg is required for URI mode. Install with: uv add 'psycopg[binary]>=3.2.0'") from exc

	with psycopg.connect(db_uri, connect_timeout=10) as conn:
		with conn.cursor() as cur:
			cur.execute("select 1")
			row = cur.fetchone()
			return bool(row and row[0] == 1)


def test_connection() -> bool:
	"""Auto-select connection mode: URI first, then REST."""
	if os.getenv("SUPABSE_DATABASE_URL"):
		return test_postgres_uri_connection()
	return test_supabase_connection()


if __name__ == "__main__":
	try:
		ok = test_connection()
		mode = "Postgres URI" if os.getenv("SUPABSE_DATABASE_URL") else "Supabase REST"
		print(f"{mode} connection successful." if ok else f"{mode} connection check failed.")
	except Exception as exc:
		print(f"Supabase connection error: {exc}")
