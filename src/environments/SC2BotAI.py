import sc2


class SC2BotAIBase(sc2.BotAI):
    """
        An abstract class for interfacing RL agent and sc2.BotAI
    """

    def __init__(self):
        super(SC2BotAIBase, self).__init__()

    async def on_step(self, action_args):
        """
        GET action_args from RL agent side and process the input to the SC2 commands!

        #expected behaviour#
        if action_args is None: # for supporting non-controllable states.
            action_list = []
        else:
            action_list = processing_inputs(action_args)

        await self.do_actions(action_list)

        """

        raise NotImplementedError


class SimpleSC2BotAI(SC2BotAIBase):
    """
        an dumb agent that do nothing
    """

    async def on_step(self, action_list=None):
        await self.do_actions(action_list)
