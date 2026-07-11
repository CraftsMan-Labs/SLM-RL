"""Per-game RAM decoders. Each module exports named offset constants plus a
`decode(ram) -> dict` returning only the variables it could verify
empirically against the running ALE env — never guessed."""
