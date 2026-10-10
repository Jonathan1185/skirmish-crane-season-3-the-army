"""Role-based Crane agent.

Every unit owns one instance of ``Agent``. The units therefore coordinate from the
allies and enemies they can currently see, rather than through shared Python state.
"""

from sandbox.crane import action, me, paths, tile, visible
from sandbox.observation_types import AxialPosition, SkirmishAction, SkirmishObservation


class Agent:
    """Keeps archers safe, sends cavalry after enemies, and assigns footmen as guards."""

    THREAT_DISTANCE = 3

    def reset(self, seed, observation) -> None:
        # Health history lets a footman notice a visible archer was hurt between activations.
        # The observation does not identify the attacker, so this is the closest available signal.
        self._archer_health: dict[str, int] = {}

    def act(self, observation: SkirmishObservation) -> SkirmishAction:
        unit_type = me.unit_type(observation)

        if unit_type == "archer":
            return self._archer_order(observation)
        if unit_type == "cavalry":
            return self._cavalry_order(observation)
        return self._footman_order(observation)

    def _archer_order(self, observation: SkirmishObservation) -> SkirmishAction:
        """Retreat as far as possible from the closest enemy after the whole walk."""
        enemies = visible.enemies(observation)
        footmen = [ally for ally in visible.allies(observation) if ally["type"] == "footman"]
        if not enemies:
            if footmen:
                here = me.position(observation)
                footman = min(footmen, key=lambda ally: tile.distance(here, ally["position"]))
                return self._move_toward(observation, footman["position"])
            return self._move_toward(observation, tile.at_center(observation))

        # Score every complete legal route by its final tile. The minimum matters because an
        # archer is only as safe as its nearest enemy. Once routes are equally safe, the archer
        # moves closer to its nearest footman, then spends any remaining movement it can.
        here = me.position(observation)
        path_id = max(
            action.legal_paths(observation),
            key=lambda path: (
                min(
                    tile.distance(tile.at_path_end(here, path), enemy["position"])
                    for enemy in enemies
                ),
                -min(
                    (tile.distance(tile.at_path_end(here, path), footman["position"]) for footman in footmen),
                    default=0,
                ),
                len(paths.decode(path)),
                -path,
            ),
        )
        landing = tile.at_path_end(here, path_id)
        target = min(enemies, key=lambda enemy: tile.distance(landing, enemy["position"]))
        return action.move(path_id, target["unit_id"], observation)

    def _cavalry_order(self, observation: SkirmishObservation) -> SkirmishAction:
        """Use the full legal route that gets closest to the nearest visible enemy."""
        enemies = visible.enemies(observation)
        if not enemies:
            return self._move_toward(observation, tile.at_center(observation))

        here = me.position(observation)
        target = min(enemies, key=lambda enemy: tile.distance(here, enemy["position"]))
        return self._move_toward(observation, target["position"], target["unit_id"])

    def _footman_order(self, observation: SkirmishObservation) -> SkirmishAction:
        """Guard a visible archer, intercepting enemies that threaten it first."""
        archers = [ally for ally in visible.allies(observation) if ally["type"] == "archer"]
        enemies = visible.enemies(observation)

        # The roster does not mark dead units, and units outside vision are also absent from the
        # observation. Once no archer is visible, there is no guard position to maintain; this
        # covers a dead archer and turns the footman into an attacker against the weakest enemy it
        # can legally name.
        if not archers and enemies:
            here = me.position(observation)
            target = min(
                enemies,
                key=lambda enemy: (enemy["hit_points"], tile.distance(here, enemy["position"])),
            )
            return self._move_toward(observation, target["position"], target["unit_id"])

        hurt_archers = [
            archer
            for archer in archers
            if archer["unit_id"] in self._archer_health
            and archer["hit_points"] < self._archer_health[archer["unit_id"]]
        ]
        self._archer_health.update({archer["unit_id"]: archer["hit_points"] for archer in archers})

        # The game reports no explicit "attacked by" event. A health drop reveals that an archer
        # was attacked, while an enemy within three hexes is an immediate visible threat.
        threatened_archers = [
            archer
            for archer in archers
            if archer in hurt_archers
            or any(tile.distance(enemy["position"], archer["position"]) <= self.THREAT_DISTANCE for enemy in enemies)
        ]
        threats = [
            enemy
            for enemy in enemies
            if any(
                archer in hurt_archers
                or tile.distance(enemy["position"], archer["position"]) <= self.THREAT_DISTANCE
                for archer in threatened_archers
            )
        ]
        if threats:
            here = me.position(observation)
            # Prefer the enemy nearest the threatened archer, then the one this footman reaches
            # soonest. If the attacker is not visible, the API cannot identify it exactly.
            target = min(
                threats,
                key=lambda enemy: (
                    min(
                        tile.distance(enemy["position"], archer["position"])
                        for archer in threatened_archers
                    ),
                    tile.distance(here, enemy["position"]),
                ),
            )
            return self._move_toward(observation, target["position"], target["unit_id"])

        if archers:
            here = me.position(observation)
            # The closest archer is the one this footman can protect most reliably.
            archer = min(archers, key=lambda ally: tile.distance(here, ally["position"]))
            return self._move_toward(observation, archer["position"])

        # An archer outside a footman's vision has no known position. Move to the shared center
        # until it becomes visible again instead of chasing stale information.
        return self._move_toward(observation, tile.at_center(observation))

    def _move_toward(
        self,
        observation: SkirmishObservation,
        goal: AxialPosition,
        target_id: str | None = None,
    ) -> SkirmishAction:
        """Choose the legal path whose endpoint is closest to ``goal``.

        If several paths end equally close, prefer the route with more steps. This uses each
        unit's available movement points while never trading progress toward its current goal
        merely to walk farther.
        """
        here = me.position(observation)
        path_id = min(
            action.legal_paths(observation),
            key=lambda path: (
                tile.distance(tile.at_path_end(here, path), goal),
                -len(paths.decode(path)),
                path,
            ),
        )
        return action.move(path_id, target_id, observation) if target_id else action.move(path_id)
