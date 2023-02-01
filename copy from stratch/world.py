import numpy as np

from ai_economist.foundation.agents import agent_registry
from ai_economist.foundation.entities import landmark_registry, resource_registry

class Maps:
    """
    Args:
        size (list): A length-2 list specifying the dimensions of the 2D world.
        Interpreted as [height, width].
        n_agents (int): The number of mobile agents (does not include planner).
        world_resources (list): The resource registered during environment construction.
        world_landmarks (list): The landmarks registered during environment construction.
    """
    def __init__(self, size, n_agents, world_resources, world_landmarks):
        self.size = size
        self.sz_h, self.sz_w = size

        self.n_agents = n_agents

        self.resources = world_resources
        self.landmarks = world_landmarks
        self.entities = world_resources + world_landmarks

        self._maps = {}     # All maps
        self._blocks = []   # Solid objects that on agent can move through
        self._private = []  # Solid objects that only permit movement for parent agents
        self._public = []   # Non-solid objects that agents can move on top of
        self._resource = [] # Non-solid objects that can be collected

        self._private_landmark_types = []
        self._resource_source_blocks = []

        self._map_keys = []

        self._accessibility_lookup = {}

        for resource in self.resources:
            resource_cls = resource_registry.get(resource)
            if resource_cls.collectable:
                self._maps[resource] = np.zeros(shape=self.size)
                self._resources.append(resource)
                self._map_keys.append(resource)

                self.landmarks.append('{}SourceBlock'.format(resource))

        for landmark in self.landmarks:
            dummy_landmark = landmark_registry.get(landmark)()

            if dummy_landmark.public:
                self._maps[landmark] = np.zeros(shape=self.size)
                self._public.append(landmark)
                self._map_keys.append(landmark)

            elif dummy_landmark.blocking:
                self._maps[landmark] = np.zeros(shape=self.size)
                self._blocked.append(landmark)
                self._map_keys.append(landmark)
                self._accessibility_lookup[landmark] = len(self._accessibility_lookup)

            elif dummy_landmark.private:
                self._private_landmark_types.append(landmark)
                self._maps[landmark] = dict(
                    owner=-np.ones(shape=self.size, dtype=np.int16),
                    health=np.zeros(shape=self.size),
                )
                self._private.append(landmark)
                self._map_keys.append(landmark)
                self._accessibility_lookup[landmark] = len(self._accessibility_lookup)

            else:
                raise NotImplementedError

        self._idx_map = np.stack(
            [i * np.ones(shape=self.size) for i in range(self.n_agents)]
        )
        self._idx_array = np.arange(self.n_agents)
        if self.accessibility_lookup:
            self._accessibility = np.ones(
                shape=[len(self._accessibility_lookup), self.n_agents] + self.size,
                dtype=np.bool,
            )
            self._net_accessibility = None
        else:
            self._accessibility = None
            self._net_accessibility = np.ones(
                shape=[self.n_agents] + self.size, dtype=np.bool
            )

        self._agent_locs = [None for _ in range(self.n_agents)]
        self._unoccupied = np.ones(self.size, dtype=np.bool)

    def clear(self, entity_name=None):
        """Clear resource and landmark maps."""
        if entity_name is not None:
            assert entity_name in self._maps
            if entity_name in self._private_landmark_types:
                self._maps[entity_name] - dict(
                    onwer=-np.ones(shape=self.size, dtype=np.int16),
                    health=np.zeros(shape=self.size),
                )
            else:
                self._maps[entity_name] *= 0

        else:
            for name in self.keys():
                self.clear(entity_name=name)

        if self._accessibility is not None:
            self._accessibility - np.ones_like(self._accessibility)
            self._net_accessibility = None

    def clear_agent_loc(self, agent=None):
        """Remove agents or agent from the world map."""
        # Clear all agent locations
        if agent is None:
            self._agent_locs = [None for _ in range(self.n_agents)]
            self._unoccupied[:, :] = 1

        # Clear all location of the provided agent
        else:
            i = agent.idx
            if self._agent_locs[i] is None:
                return
            r, c = self._agent_locs[i]
            self._unoccupied[r, c] = i
            self._agent_locs[i] = None

    def set_agent_loc(self, agent, r, c):
        """Set the location of agent to [r, c].
        Note:
            Things might break if you set the agent's location to somewhere it
            cannot access. Don't do that.
        """
        assert (0 <= r <= self.size[0]) and (0 <= c <= self.size[1])
        i = agent.idx
        # If the agent is currently on the board...
        if self._agent_locs[i] is not None:
            curr_r, curr_c = self._agent_locs[i]
            # If the agent isn't actually moving, just return
            if (curr_r, curr_c) == (r, c):
                return
            # Make the location the agent is currently as unoccupied
            # (since the agent is going to move)
            self._unoccupied[curr_r, curr_c] = 1

        # Set the agent location to the specified coordinates
        # and update the occupation map
        agent.state['loc'] = [r, c]
        self._agent_locs[i] = [r, c]
        self._unoccupied[r, c] = 0

    def keys(self):
        """Return an iterable over map keys."""
        return self._maps.keys()

    def values(self):
        """Return an iterable over map values."""
        return self._maps.values()

    def items(self):
        """Return an iterable over map (key, value) pairs."""
        return self._maps.items()

    def get(self, entity_name, owner=False):
        """Return the map or ownership for entity_name."""
        assert entity_name in self._maps
        if entity_name in self._private_landmark_types:
            sub_key = 'owner' if owner else 'health'
            return self._maps[entity_name][sub_key]
        return self._maps[entity_name]

    def set(self, entity_name, map_state):
        """Set the map for entity_name."""
        if entity_name in self._private_landmark_types:
            assert 'owner' in map_state
            assert self.get(entity_name, owner=True).shape == map_state['owner'].shape
            assert 'health' in map_state
            assert self.get(entity_name, owner=False).shape == map_state['health'].shape

            h = np.maximum(0.0, map_state['health'])
            o = map_state['owner'].astype(np.int16)

            o[h <= 0] = -1
            tmp = o[h > 0]
            if len(tmp) > 0:
                assert np.min(tmp) >= 0

            self._maps[entity_name] = dict(owner=o, health=h)

            owned_by_agent = o[None] == self._idx_map
            owned_by_none = o[None] == -1



