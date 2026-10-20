# Planned System Architecture

**Status:** Proposed structure. The repository layout does not indicate completed implementation.

## Technical workstreams

1. **Crop recommendation** (`01-crop-recommendation/`): approved data references, preprocessing, model experimentation and evaluation, and recommendation logic.
2. **Crop health diagnosis** (`02-crop-health/`): permitted image data, preprocessing, computer vision model experimentation and evaluation, and prediction logic.

## Shared integration layer

The planned `03-shared/` components include a dashboard for interacting with both workstreams, shared APIs that establish input/output contracts, and integration logic that connects workstream outputs to shared services. Interfaces and implementation choices require team agreement and validation.

## Supporting data and documentation

`04-data-reporting/` centralizes data provenance and quality, experiments, and report drafts. `docs/api_contracts.md` records proposed contracts; `docs/project_plan.md` tracks assignments and risks.

## Planned data flow (conceptual)

Approved inputs -> workstream-specific validation and processing -> evaluated workstream model/logic (once developed) -> agreed API/integration layer -> dashboard output.

**Open decisions:** dataset approvals, concrete model approaches, interface specifications, deployment architecture, and stakeholder validation. None is asserted as complete here.
