# Synthetic MeTTa-like directives for the starter gym.
# Lines beginning with ! are parser directives, not real MeTTa syntax.
!exec rule=investigate before=trust:1,evidence:0,heat:1,scene:2 after=trust:1,evidence:1,heat:2,scene:3 invariants=bounds,scene_progress
!exec rule=defuse before=trust:1,evidence:2,heat:4,scene:3 after=trust:1,evidence:2,heat:2,scene:4 invariants=bounds,heat_reduced
!exec rule=arith_add_heat before=heat:1,delta:2 after=heat:3,delta:2 invariants=arithmetic,bounds
!exec rule=compare_heat_gate before=heat:4,threshold:3 after=heat:4,threshold:3,gate:1 invariants=comparison,boolean_gate
!exec rule=precondition_fail_defuse before=heat:1,scene:2 after=heat:1,scene:2 invariants=precondition_failed,no_state_change passed=false
!exec rule=postcondition_update_scene before=scene:2,complete:0 after=scene:3,complete:1 invariants=postcondition_update,scene_progress
!deduce rule=type_check_action before=action:{befriend,defuse,42};type:{Action,Number} after=action:{befriend,defuse};type:{Action} eliminated=action:42,type:Number conflict=false soundness=env_sound_dead
!deduce rule=deduction_closure before=route:{ally,solo,rush};heat:{0,1,2,3};safe:{yes,no} after=route:{ally,solo};heat:{0,1,2};safe:{yes} eliminated=route:rush,heat:3,safe:no conflict=false soundness=env_sound_dead
!deduce rule=bottom_conflict before=choice:{left,right};requires:{key};has_key:{no} after=choice:{};requires:{key};has_key:{no} eliminated=choice:left,choice:right conflict=true soundness=env_sound_dead
!deduce rule=ending_requires_low_heat before=secret:{possible,dead};heat:{3,4};scene:{4,5} after=secret:{dead};heat:{3,4};scene:{4,5} eliminated=secret:possible conflict=true soundness=env_sound_dead
!deduce rule=replay_prune_bad_route before=path:{ally,solo,rush};skill:{befriend,defuse,rush} after=path:{ally,solo};skill:{befriend,defuse} eliminated=path:rush,skill:rush conflict=false soundness=experience_sound_dead
!route task=secret_ending candidates=befriend,investigate,defuse,rush chosen=defuse reason=heat_too_high
!route task=typed_arithmetic candidates=arith_add_heat,compare_heat_gate,type_check_action chosen=arith_add_heat reason=integer_state_update
!route task=failed_precondition candidates=precondition_fail_defuse,postcondition_update_scene,deduction_closure chosen=precondition_fail_defuse reason=guard_blocks_transition
!repair diagnostic=unknown_symbol buggy="(call rush-fast state)" repaired="(call rush state)" passed=true
!repair diagnostic=malformed_rule buggy="(= (can-defuse $s) (and (Heat $s $h) (< $h)))" repaired="(= (can-defuse $s) (and (Heat $s $h) (> $h 1)))" passed=true
