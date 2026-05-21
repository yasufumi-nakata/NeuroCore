## Summary

- 

## Safety and Runtime Boundary

- [ ] Decoder weights, checkpoints, and private EEG data are not included.
- [ ] OS input and AI-agent execution still happen outside `backend/neurocore`.
- [ ] New allowed behavior has a matching blocked-path test when safety-relevant.

## Checks

- [ ] `python -m pytest`
- [ ] `python -m neurocore.cli self-test --json`
- [ ] `python -m ruff check backend scripts`
- [ ] `cd frontend && npm run build`
- [ ] `python scripts/oss_health_check.py`
