from research_gym.scripts import audit_rlm_evidence_acquisition_v0 as audit


def test_sealed_evidence_acquisition_audit_recomputes() -> None:
    result = audit.build_audit(audit.DEFAULT_OUTPUT)
    fixed = result["task_level_contrasts"]["mesh_fixed_no_query"]
    assert result["canonical_artifact_replay"]["summary_exact"]
    assert result["canonical_artifact_replay"]["comparisons_exact"]
    assert result["canonical_artifact_replay"]["task_shards_replayed"] == 24
    assert result["canonical_artifact_replay"]["evidence_receipt_failures"] == 0
    assert result["hard_gates"]["typed_unsafe_count"] == 0
    assert result["hard_gates"]["forced_failure_no_op_passed"]
    assert fixed["net_sign_counts"] == {"positive": 2, "zero": 14, "negative": 8}
    assert result["query_policy"]["no_query_tasks"] == 14
    assert result["query_policy"]["oracle_sequence_matches"] == 10
    assert result["query_policy"]["query_id_counts"].get("counterfactual_rollout", 0) == 0
