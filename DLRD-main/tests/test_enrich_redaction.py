"""Tests for app.enrich.sanitize_for_enrichment — credential redaction applied
to the LLM enrichment payload (defense-in-depth on the egress path).

Self-contained: run directly with `python tests/test_enrich_redaction.py` or via
`pytest tests/`. The fixture is a .sh-style wrapper around bteq, mixing real
credential lines with benign content (including a PWD= working-directory var
that must NOT be treated as a password).
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.enrich import sanitize_for_enrichment  # noqa: E402

FIXTURE = """#!/bin/sh
# nightly load wrapper
export DB_HOST=prod-db.internal
cd "$HOME/jobs"
bteq <<'EOF'
.LOGON tdpid/myuser,SECRETpw123
DELETE FROM stg.t ALL;
.LOGOFF
EOF
    .logon  prodtd/svc_acct,Pa$$w0rd
.CONNECT dbc/admin,Adm1nPass
export PASSWORD=hunter2
CREATE USER analyst IDENTIFIED BY 'Td2SecretPass' AS PERM=1e9;
GRANT SELECT ON db TO svc IDENTIFIED BY UnquotedPw99 AS PERM=500;
conn = "https://dbuser:SuperUrlPw@db.host:5432/app?sslmode=require"
git clone git@github.com:org/repo.git
psql -h db-host:5432 -U reader
export TOKEN=ghp_AbC123tokenXYZ
api_key=AKIAFAKEKEY123abc
export SECRET_KEY=topSecretKeyVal
secret_sauce=ketchup
DB_PASSWORD=dbSecret123
export TD_PASSWORD=tdSecretPw
ETL_PW=etlPw789
PW=barePw42
MDP=motdepasse1
MOT_DE_PASSE=frenchPass2
PASSPHRASE=myPhrase5
BYPASS=true
COMPASS=42
curl -H "Authorization: Bearer abc123BearerTok" https://api.example.com/v1
# remember to rotate the database password every quarter
PWD=/some/path
echo "cwd is $PWD"
"""

SECRETS = [
    "SECRETpw123", "Pa$$w0rd", "Adm1nPass", "hunter2",                 # 626c8c8 patterns
    "Td2SecretPass", "UnquotedPw99",                                   # IDENTIFIED BY
    "SuperUrlPw",                                                      # URL basic-auth
    "ghp_AbC123tokenXYZ", "AKIAFAKEKEY123abc", "topSecretKeyVal",      # token / api_key / secret_key
    "abc123BearerTok",                                                 # Bearer
    "dbSecret123", "tdSecretPw", "etlPw789",                           # prefixed (DB_/TD_/ETL_)
    "barePw42", "motdepasse1", "frenchPass2", "myPhrase5",             # short PW + French + passphrase
]


def test_secrets_are_gone():
    out = sanitize_for_enrichment(FIXTURE)
    for s in SECRETS:
        assert s not in out, f"secret leaked: {s}"


def test_keywords_remain_redacted():
    out = sanitize_for_enrichment(FIXTURE)
    assert ".LOGON <redacted>" in out           # keyword kept, credential gone
    assert ".CONNECT <redacted>" in out
    assert "PASSWORD=<redacted>" in out
    assert "export PASSWORD=<redacted>" in out   # the `export ` prefix survives


def test_indented_lowercase_logon_preserved():
    out = sanitize_for_enrichment(FIXTURE)
    # leading whitespace + original (lowercase) keyword preserved
    assert "    .logon <redacted>" in out


def test_pwd_is_not_a_password():
    out = sanitize_for_enrichment(FIXTURE)
    assert "PWD=/some/path" in out               # working-dir var untouched
    assert 'echo "cwd is $PWD"' in out           # $PWD reference untouched


def test_surrounding_lines_unchanged():
    out = sanitize_for_enrichment(FIXTURE)
    assert "export DB_HOST=prod-db.internal" in out
    assert "DELETE FROM stg.t ALL;" in out
    assert 'cd "$HOME/jobs"' in out
    assert ".LOGOFF" in out and ".LOGOFF <redacted>" not in out  # not a credential command


def test_identified_by_redacted():
    out = sanitize_for_enrichment(FIXTURE)
    assert out.count("IDENTIFIED BY <redacted>") == 2     # quoted and unquoted forms
    assert "AS PERM=1e9;" in out and "AS PERM=500;" in out  # surrounding content preserved


def test_url_basic_auth_password_redacted():
    out = sanitize_for_enrichment(FIXTURE)
    # only the password between : and @ is redacted; user + @host:port/path kept
    assert "https://dbuser:<redacted>@db.host:5432/app?sslmode=require" in out


def test_token_like_assignments_redacted():
    out = sanitize_for_enrichment(FIXTURE)
    assert "export TOKEN=<redacted>" in out
    assert "api_key=<redacted>" in out
    assert "export SECRET_KEY=<redacted>" in out


def test_bearer_redacted():
    out = sanitize_for_enrichment(FIXTURE)
    assert 'Authorization: Bearer <redacted>"' in out      # token gone, closing quote kept


def test_benign_lookalikes_untouched():
    out = sanitize_for_enrichment(FIXTURE)
    assert "git@github.com:org/repo.git" in out            # no ://…:…@  -> not basic-auth
    assert "db-host:5432" in out                           # host:port, no credentials
    assert "secret_sauce=ketchup" in out                   # \\b…= guard: not a secret name
    assert "# remember to rotate the database password every quarter" in out  # prose
    assert "PWD=/some/path" in out                         # cwd var, never a password
    assert "https://api.example.com/v1" in out             # credential-free URL untouched


def test_prefixed_credential_vars_redacted():
    out = sanitize_for_enrichment(FIXTURE)
    assert "DB_PASSWORD=<redacted>" in out               # prefix before the keyword
    assert "export TD_PASSWORD=<redacted>" in out        # export prefix survives too
    assert "ETL_PW=<redacted>" in out


def test_short_and_french_credential_vars_redacted():
    lines = sanitize_for_enrichment(FIXTURE).splitlines()
    assert "PW=<redacted>" in lines                       # bare short form
    assert "MDP=<redacted>" in lines                      # French abbreviation
    assert "MOT_DE_PASSE=<redacted>" in lines             # French, underscored
    assert "PASSPHRASE=<redacted>" in lines


def test_credential_var_lookalikes_untouched():
    out = sanitize_for_enrichment(FIXTURE)
    assert "BYPASS=true" in out                           # PASS not suffix-matched
    assert "COMPASS=42" in out
    assert "PWD=/some/path" in out                        # PW followed by D, not '='


def test_line_structure_preserved():
    out = sanitize_for_enrichment(FIXTURE)
    assert out.count("\n") == FIXTURE.count("\n")  # redaction is in-line, no lines added/removed


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {t.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run() else 0)
