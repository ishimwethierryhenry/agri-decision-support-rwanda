# API Contract Documentation Template

**Status:** Template only. No implemented API endpoints are asserted.

Create a section for each **proposed or implemented** interface, using the following fields:

| Field | What to document |
| --- | --- |
| Interface name | Descriptive, unique name |
| Status | Proposed / agreed / implemented / deprecated |
| Responsible workstream | Crop recommendation / crop health / shared |
| Purpose | Intended behavior and user need |
| Method and route | HTTP verb and path, only once agreed |
| Input fields | Name, type, required/optional, units, valid range |
| Output fields | Name, type, meaning, uncertainty or confidence if applicable |
| Validation | Missing, malformed, or out-of-range input behavior |
| Error responses | Status categories, message shape, recoverable cases |
| Data/privacy | Sensitive fields, retention, access constraints |
| Contract version | Version, reviewer, and last reviewed date when known |
| Test coverage | Linked contract and integration tests when written |

## Review checklist

Confirm that field naming, units, error behavior, and confidence semantics match the workstream implementation before marking a contract as implemented. Do not invent endpoints or sample responses that appear production-ready.
