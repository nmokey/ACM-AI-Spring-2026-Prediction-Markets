---
name: Never read .env file
description: User has explicitly forbidden reading their .env file under any circumstances
type: feedback
---

Never read, open, cat, or inspect the user's .env file under any circumstances.

**Why:** User explicitly requested this as a permanent rule.

**How to apply:** If a task involves environment variables, ask the user to confirm values are set or check .env.example instead. Never use Read, Bash cat, or any other tool on .env files.
