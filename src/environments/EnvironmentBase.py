import asyncio
import logging
from asyncio import new_event_loop as loop

import sc2
from sc2.protocol import ConnectionAlreadyClosed
from sc2.game_state import GameState
from sc2.sc2process import SC2Process
from sc2.main import _setup_host_game
from sc2.player import Bot, Human


class NotCloseSC2Process(SC2Process):

    def __init__(self, *args, **kwargs):
        super(NotCloseSC2Process, self).__init__(*args, **kwargs)

    async def __aexit__(self, *args):
        pass


class SC2EnvironmentBase:
    """
        Reinforcement Learning friendly python-sc2 wrapper
    """

    def __init__(self, map_name, allies, realtime=False, frame_skip_rate=1):
        self.name = map_name
        self.map = sc2.maps.get(map_name)
        self.allies = allies
        self.players = [self.allies]
        self.realtime = realtime

        self.host_server = None
        self.join_server = None

        self.allies_client = None
        self.allies_id = None
        self.allies_server = None

        self.allies_game_data = None
        self.allies_game_info = None
        self.allies_game_state = None

        self.frame_skip_rate = frame_skip_rate

        self.t = 0
        _ = self._reset()

    def close(self):
        if self.allies_client is not None:
            self._loop_run(self.allies_client.leave())
            self._loop_run(self.allies_client.quit())

        if self.host_server is not None:
            self.loop.run_until_complete(self.host_server._Controller__process._close_connection())
            self.host_server._Controller__process._clean()

        if self.join_server is not None:
            self._loop_run(self.join_server._Controller__process._session.close())
            self.join_server._Controller__process._clean()

        if self.allies_game_data is not None:
            self.allies_game_data = None

    # Wrapper for python-sc2 main._setup_host_game method
    async def _setup_host_game(self, server, **kwargs):
        client = await _setup_host_game(server,
                                        map_settings=self.map,
                                        players=self.players,
                                        realtime=self.realtime,
                                        **kwargs)
        return client

    # Overload existing python-sc2 main._host_game method not to act by default!
    async def _host_game(self, portconfig=None, save_replay_as=None):
        #assert self.num_players > 0, "Can't create a game without players"

        # At least 1 not computer player is required!
        assert any(isinstance(p, (Human, Bot)) for p in self.players)

        async with NotCloseSC2Process() as server:
            await server.ping()
            client = await self._setup_host_game(server)

            try:
                player_id = await client.join_game(race=self.allies.race, portconfig=portconfig)
                logging.info(f"Player id: {player_id}")

                if save_replay_as is not None:
                    await client.save_replay(save_replay_as)

            except ConnectionAlreadyClosed:
                logging.error(f"Connection was closed before the game ended")
                return None

        self.host_server = server

        return [client, player_id, server]

    async def run_game(self):
        result = await self._host_game()
        return result

    def _ai_preparation(self, bot_ai, player_id, client):
        game_data = self._loop_run(client.get_game_data())
        game_info = self._loop_run(client.get_game_info())
        state = self._loop_run(client.observation())
        game_state = GameState(state.observation, game_data)

        bot_ai._prepare_start(client, player_id, game_info, game_data)
        bot_ai.on_start()
        bot_ai._prepare_step(game_state)
        bot_ai._prepare_first_step()
        return game_data, game_info, game_state

    def _loop_run(self, coroutine):
        return self.loop.run_until_complete(coroutine)

    def _reset(self):
        self.close()
        self.loop = loop()
        asyncio.set_event_loop(self.loop)
        self.t = 0

        allies_info = self._loop_run(self.run_game())
        self.allies_client = allies_info[0]
        self.allies_id = allies_info[1]
        self.allies_server = allies_info[2]

        ret_allies = self._ai_preparation(bot_ai=self.allies.ai, player_id=self.allies_id,
                                          client=self.allies_client)
        self.allies_game_data = ret_allies[0]
        self.allies_game_info = ret_allies[1]
        self.allies_game_state = ret_allies[2]
        return self.allies_game_state


    def _observe(self):
        return self.allies_game_state

    def _step(self, action_args):
        # if self.env.allies_client._status

        self.t += 1
        allies_game_state = self.allies_game_state

        # current game_state
        self._loop_run(self.allies.ai.issue_events())
        self._loop_run(self.allies.ai.on_step(action_args))
        for _ in range(self.frame_skip_rate):
            self._loop_run(self.allies_client.step())

        allies_state = self._loop_run(self.allies_client.observation())
        next_game_state = GameState(allies_state.observation, self.allies_game_data)
        self.allies_game_state = next_game_state
        self.allies.ai._prepare_step(allies_game_state)

        game_result = self.allies_client._game_result
        done = True if game_result else False

        return next_game_state, done

    def reset(self):
        raise NotImplementedError("This method should be implemented in the child class")

    def observe(self):
        raise NotImplementedError("This method should be implemented in the child class")

    def step(self, *args, **kwargs):
        raise NotImplementedError("This method should be implemented in the child class")