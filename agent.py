"""Role-based Crane agent with terrain-aware movement.

Every unit owns one instance of ``Agent``. The units therefore coordinate from the
allies and enemies they can currently see, rather than through shared Python state.
"""

from heapq import heappop, heappush

from sandbox.crane import action, me, paths, tile, units, visible
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
        """Search for targets, then kite the closest one at maximum arrow range."""
        enemies = visible.enemies(observation)
        footmen = [ally for ally in visible.allies(observation) if ally["type"] == "footman"]
        if not enemies:
            # The center gives the archer the best chance to bring its long vision to bear. A
            # footman heading toward this archer will still meet it there or on the route.
            return self._move_toward(observation, tile.at_center(observation))

        here = me.position(observation)
        target = min(enemies, key=lambda enemy: tile.distance(here, enemy["position"]))
        attack_range = units.STATS["archer"].attack_range

        # Maximize distance from the closest target while keeping it in arrow range: the preferred
        # endpoint is exactly range 6, where the archer attacks without closing in. A hill can
        # extend vision beyond attack range, in which case the archer instead paths toward sight.
        attack_paths = [
            path
            for path in action.legal_paths(observation)
            if tile.distance(tile.at_path_end(here, path), target["position"]) <= attack_range
        ]
        if not attack_paths:
            return self._move_toward(observation, target["position"], target["unit_id"])
        path_id = max(
            attack_paths,
            key=lambda path: (
                tile.distance(tile.at_path_end(here, path), target["position"]),
                min(tile.distance(tile.at_path_end(here, path), enemy["position"]) for enemy in enemies),
                -min(
                    (tile.distance(tile.at_path_end(here, path), footman["position"]) for footman in footmen),
                    default=0,
                ),
                len(paths.decode(path)),
                -path,
            ),
        )
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
        path_id = self._pathfind_path(observation, goal)
        if path_id is None:
            # A hidden unit can block a route that was planned from the visible state. Fall back
            # to the best currently legal endpoint; the next activation replans from there.
            path_id = min(
                action.legal_paths(observation),
                key=lambda path: (
                    tile.distance(tile.at_path_end(here, path), goal),
                    -len(paths.decode(path)),
                    path,
                ),
            )
        return action.move(path_id, target_id, observation) if target_id else action.move(path_id)

    def _pathfind_path(self, observation: SkirmishObservation, goal: AxialPosition) -> int | None:
        """Return the longest legal prefix of an A* route to ``goal``."""
        route = self._a_star_route(observation, me.position(observation), goal)
        if route is None:
            return None

        legal_paths = set(action.legal_paths(observation))
        for length in range(min(paths.MAX_STEPS, len(route)), 0, -1):
            path_id = paths.encode(route[:length])
            if path_id in legal_paths:
                return path_id
        return 0

    def _a_star_route(
        self, observation: SkirmishObservation, start: AxialPosition, goal: AxialPosition
    ) -> tuple[int, ...] | None:
        """Plan through known passable tiles using the game's terrain movement costs."""
        visible_units = [*visible.allies(observation), *visible.enemies(observation)]
        occupied = {(unit["position"]["q"], unit["position"]["r"]) for unit in visible_units}
        start_key = (start["q"], start["r"])
        goal_key = (goal["q"], goal["r"])

        # A visible unit occupies its tile, so reach one of its free neighboring tiles instead.
        if goal_key in occupied or not self._is_passable(observation, goal):
            destinations = {
                (neighbor["q"], neighbor["r"])
                for neighbor in tile.neighbors(goal).values()
                if self._is_passable(observation, neighbor)
                and (neighbor["q"], neighbor["r"]) not in occupied
            }
        else:
            destinations = {goal_key}
        if not destinations:
            return None

        def heuristic(position: tuple[int, int]) -> int:
            current: AxialPosition = {"q": position[0], "r": position[1]}
            return min(tile.distance(current, {"q": q, "r": r}) for q, r in destinations)

        frontier: list[tuple[int, int, int, tuple[int, int]]] = [(heuristic(start_key), 0, 0, start_key)]
        costs = {start_key: 0}
        previous: dict[tuple[int, int], tuple[tuple[int, int], int]] = {}
        sequence = 0

        while frontier:
            _, cost_so_far, _, current = heappop(frontier)
            if cost_so_far != costs.get(current):
                continue
            if current in destinations:
                route: list[int] = []
                while current != start_key:
                    current, direction = previous[current]
                    route.append(direction)
                return tuple(reversed(route))

            current_position: AxialPosition = {"q": current[0], "r": current[1]}
            for direction, neighbor in tile.neighbors(current_position).items():
                neighbor_key = (neighbor["q"], neighbor["r"])
                if neighbor_key in occupied or not self._is_passable(observation, neighbor):
                    continue
                next_cost = cost_so_far + self._terrain_cost(observation, neighbor)
                if next_cost >= costs.get(neighbor_key, float("inf")):
                    continue
                costs[neighbor_key] = next_cost
                previous[neighbor_key] = (current, direction)
                sequence += 1
                heappush(
                    frontier,
                    (next_cost + heuristic(neighbor_key), next_cost, sequence, neighbor_key),
                )
        return None

    @staticmethod
    def _is_passable(observation: SkirmishObservation, position: AxialPosition) -> bool:
        return tile.terrain_at(observation, position)["terrain"] not in {"water", "void"}

    @staticmethod
    def _terrain_cost(observation: SkirmishObservation, position: AxialPosition) -> int:
        """Mirror the movement-point cost used by the game for a passable tile."""
        ground = tile.terrain_at(observation, position)
        terrain_cost = 2 if ground["terrain"] == "hill" else 1
        feature_cost = {"forest": 1, "marsh": 2}.get(ground["feature"], 0)
        return terrain_cost + feature_cost
