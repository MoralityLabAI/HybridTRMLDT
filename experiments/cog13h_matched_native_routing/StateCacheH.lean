import Lean

open Lean Elab Tactic Meta

namespace COG13H

structure Request where
  id : Nat := 0
  actions : Array String := #[]
  stop : Bool := false
  deriving FromJson

structure Entry where
  actions : Array String
  state : Tactic.SavedState
  touched : Nat
  hits : Nat

private def directory : TacticM System.FilePath := do
  let some path ← IO.getEnv "COG13H_MAILBOX" | throwError "COG13H_MAILBOX missing"
  return path

private def emit (id : Nat) (j : Json) : TacticM Unit := do
  let dir ← directory
  let path := dir / s!"response-{id}.json"
  let temporary := dir / s!"response-{id}.tmp"
  IO.FS.writeFile temporary j.compress
  IO.FS.rename temporary path

private def allowed : Array String :=
  #["assumption","rfl","constructor","intro hnew","simp","simp_all","omega","linarith","ring","norm_num","positivity","aesop","exact h","all_goals assumption","all_goals omega","all_goals ring"]

private def prefixOf (xs ys : Array String) : Bool :=
  xs.size <= ys.size && xs == ys.extract 0 xs.size

private def observe : TacticM (Array String) := do
  pruneSolvedGoals
  let mut text : Array String := #[]
  for goal in ← getGoals do
    text := text.push (← ppGoal goal).pretty
  return text

private def boundedRequest (action : TacticM (Array String)) : TacticM (Array String) := do
  let heartbeats ← IO.getNumHeartbeats
  controlAt CoreM fun runInBase =>
    withReader (fun (ctx : Core.Context) =>
      { ctx with initHeartbeats := heartbeats, maxHeartbeats := 100000000 })
      (runInBase action)

private def weight (policy : String) (e : Entry) : Nat :=
  if policy == "value" then e.actions.size * (e.hits + 1) else e.touched

private def victim (policy : String) (entries : Array Entry) : Option Entry := Id.run do
  let mut best : Option Entry := none
  for e in entries do
    if let some b := best then
      if weight policy e < weight policy b ||
          (weight policy e == weight policy b && e.touched < b.touched) then
        best := some e
    else
      best := some e
  return best

private def cacheView (entries : Array Entry) : Json :=
  toJson (entries.map fun e => Json.mkObj
    [("actions", toJson e.actions), ("touched", toJson e.touched), ("hits", toJson e.hits)])

/-- Genuine tactic/term/metavariable snapshots local to one speculative theorem
    process. Cache keys are exact scripts, never pretty-printed goal strings. -/
elab "cog13h_probe" : tactic => do
  let cap := ((← IO.getEnv "COG13H_CAP").getD "0").toNat?.getD 0
  let policy := (← IO.getEnv "COG13H_POLICY").getD "lru"
  if cap > 32 || !(#["lru", "value"].contains policy) then
    throwError "invalid cache configuration"
  let dir ← directory
  let rootGoals ← observe
  let initial ← Tactic.saveState
  let mut cache : Array Entry := #[]
  emit 0 (Json.mkObj [("ready", toJson true), ("goals", toJson rootGoals)])
  let mut id : Nat := 1
  repeat
    let path := dir / s!"request-{id}.json"
    while !(← path.pathExists) do IO.sleep 5
    let raw ← IO.FS.readFile path
    let request := do
      let json ← Json.parse raw
      fromJson? json (α := Request)
    match request with
    | .error error =>
      emit id (Json.mkObj [("id", toJson id), ("protocolError", toJson error)])
    | .ok request =>
      if request.stop then break
      if request.id != id || request.actions.isEmpty || request.actions.size > 32 ||
          !(request.actions.all fun a => allowed.contains a) then
        emit id (Json.mkObj [("id", toJson id),
          ("protocolError", toJson "request id, length or grammar rejected")])
      else
        let mut chosen : Option Entry := none
        for e in cache do
          if prefixOf e.actions request.actions then
            if let some b := chosen then
              if b.actions.size < e.actions.size then chosen := some e
            else
              chosen := some e
        let mut restored : Nat := 0
        if let some e := chosen then
          e.state.restore (restoreInfo := true)
          restored := e.actions.size
          cache := (cache.filter fun b => b.actions != e.actions).push
            { e with touched := id, hits := e.hits + 1 }
        else
          initial.restore (restoreInfo := true)
        Core.resetMessageLog
        -- IO.Ref is outside the backtracked tactic state. A thrown tactic
        -- still leaves its attempted-command counter charged.
        let commands ← IO.mkRef (0 : Nat)
        let result : Except String (Array String) ←
          try
            let goals ← boundedRequest do
              for action in request.actions.extract restored request.actions.size do
                commands.modify (· + 1)
                let parsed := Parser.runParserCategory (← getEnv) `tactic action
                match parsed with
                | .error error => throwError error
                | .ok stx => evalTactic stx
              if (← Core.getMessageLog).hasErrors then throwError "tactic logged an error"
              observe
            pure (.ok goals)
          catch error =>
            pure (.error (← error.toMessageData.toString))
        let executed ← commands.get
        let mut evicted : Option (Array String) := none
        let mut inserted := false
        -- A failed request may touch an existing valid entry but never
        -- publishes a new state, even if an earlier command succeeded.
        let resultJson ←
          match result with
          | .ok goals => do
            if cap > 0 && !goals.isEmpty &&
                !(cache.any fun e => e.actions == request.actions) then
              let state ← Tactic.saveState
              cache := cache.push { actions := request.actions, state := state,
                                    touched := id, hits := 0 }
              inserted := true
              if cache.size > cap then
                if let some v := victim policy cache then
                  evicted := some v.actions
                  cache := cache.filter fun e => e.actions != v.actions
            pure (Json.mkObj [("accepted", toJson true), ("goals", toJson goals)])
          | .error error => do
            initial.restore (restoreInfo := true)
            Core.resetMessageLog
            pure (Json.mkObj [("accepted", toJson false), ("error", toJson error)])
        emit id (Json.mkObj [("id", toJson id), ("result", resultJson),
          ("restored", toJson restored), ("executed", toJson executed),
          ("inserted", toJson inserted), ("evicted", toJson evicted),
          ("cache", cacheView cache)])
    id := id + 1
  -- The speculative probe theorem is never counted as a certificate.
  initial.restore (restoreInfo := true)
  Core.resetMessageLog
  evalTactic (← `(tactic| all_goals sorry))

end COG13H
