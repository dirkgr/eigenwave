# Known Issues

## BroadcastView Tests Failing

Some BroadcastView tests are currently failing with segmentation faults after the merge of PR #1.
This needs to be investigated and fixed in a follow-up PR.

Failing tests:
- BroadcastViewTest.BroadcastCreationIsConstantTime
- BroadcastViewTest.BroadcastViewCorrectness
- BroadcastViewTest.BroadcastMaterialization
- BroadcastViewTest.CanBroadcastCheck
- BroadcastViewTest.MemoryEfficiency

These tests should be re-enabled once the issues are resolved.