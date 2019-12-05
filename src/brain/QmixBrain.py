import torch
import warnings

from src.config.ConfigBase import ConfigBase
from src.brain.Brainbase import BrainBase


class QmixBrainConfig(ConfigBase):
    def __init__(self, brain_conf=None, fit_conf=None):
        super(QmixBrainConfig, self).__init__(brain=brain_conf, fit=fit_conf)

        self.brain = {
            'prefix': 'brain',
            'optimizer': 'lookahead',
            'lr': 1e-5,
            'gamma': 1.0,
            'eps': 0.9,
            'eps_gamma': 0.995,
            'eps_min': 0.01,
            'use_double_q': False,
            'use_clipped_q': True
        }

        self.fit = {
            'prefix': 'fit',
            'tau': 0.1,
            'auto_norm_clip': False,
            'auto_norm_clip_base_val': 0.1
        }


class QmixBrain(BrainBase):
    def __init__(self, conf, qnet, mixer, qnet2, mixer2):
        super(QmixBrain, self).__init__()
        self.conf = conf
        self.brain_conf = conf.brain()
        self.fit_conf = conf.fit()

        self.use_double_q = self.brain_conf['use_double_q']
        self.use_clipped_q = self.brain_conf['use_clipped_q']

        self.gamma = self.brain_conf['gamma']

        if int(self.use_double_q) + int(self.use_clipped_q) >= 2:
            warnings.warn("Either one of 'use_double_q' or 'clipped_q' can be true. 'use_double_q' set to be false.")
            self.use_double_q = False

        if int(self.use_double_q) + int(self.use_clipped_q) == 0:
            self.use_target_q = True
        else:
            self.use_target_q = False

        self.qnet = qnet
        self.qnet2 = qnet2
        self.mixer = mixer
        self.mixer2 = mixer2

        if self.use_target:
            self.update_target_network(1.0, self.qnet, self.qnet2)
            self.update_target_network(1.0, self.mixer, self.mixer2)

        # set base optimizer
        optimizer = self.brain_conf['optimizer']
        lr = self.brain_conf['lr']
        params = list(self.qnet.parameters()) + list(self.mixer.parameters())
        self.qnet_optimizer = self.set_optimizer(target_opt=optimizer, lr=lr, params=params)

        if self.use_clipped_q:
            params = list(self.qnet2.parameters()) + list(self.mixer2.parameters())
            self.qnet2_optimizer = self.set_optimizer(target_opt=optimizer, lr=lr, params=params)

    def get_action(self, inputs, use_q1=True):

        if use_q1:
            qnet = self.qnet
        else:
            qnet = self.qnet2

        nn_actions, info_dict = qnet.get_action(**inputs)
        return nn_actions, info_dict

    @staticmethod
    def _compute_qs(inputs: dict, qnet, mixer, actions=None):
        q_dict = qnet.compute_qs(**inputs)
        qs = q_dict['qs']
        if actions is None:
            qs, _ = qs.max(dim=1)
        else:
            qs = qs.gater(-1, actions.unsqueeze(-1).long()).squeeze(dim=-1)
        q_tot = mixer.compute_qs(inputs['curr_graph'], inputs['curr_feature'], qs)
        return q_tot

    @staticmethod
    def _compute_double_qs(inputs: dict, target_qnet, target_mixer, action_qnet):
        action_q_dict = action_qnet.compute_qs(**inputs)
        action_q = action_q_dict['qs']
        actions = action_q.argmax(dim=1)

        target_q_dict = target_qnet.compute_qs(**inputs)
        target_q = target_q_dict['qs']
        target_q = target_q.gather(-1, actions.unsqueeze(-1).long()).suqeeze(dim=-1)
        target_q_tot = target_mixer(inputs['curr_graph'], inputs['curr_feature'], target_q)
        return target_q_tot

    # @staticmethod
    # def _computes_clipped_qs(inputs: dict, qnet, mixer, qnet2, mixer2):
    #     pass

    def fit(self, curr_inputs, next_inputs, actions, rewards, dones):

        q_tot = self._compute_qs(qnet=self.qnet,
                                 mixer=self.mixer,
                                 inputs=curr_inputs,
                                 actions=actions)
        if self.use_clipped_q:
            q_tot2 = self._compute_qs(qnet=self.qnet2,
                                      mixer=self.mixer2,
                                      inputs=curr_inputs,
                                      actions=actions)

        # compute next q
        with torch.no_grad():
            if self.use_double_q:
                next_q_tot = self._compute_double_qs(inputs=next_inputs,
                                                     target_qnet=self.qnet,
                                                     target_mixer=self.mixer,
                                                     action_qnet=self.qnet2)

            if self.use_clipped_q:
                next_q_tot1 = self._compute_qs(inputs=next_inputs,
                                               qnet=self.qnet,
                                               mixer=self.mixer)
                next_q_tot2 = self._compute_qs(inputs=next_inputs,
                                               qnet=self.qnet2,
                                               mixer=self.mixer2)
                next_q_tot = torch.min(next_q_tot1, next_q_tot2)

            if self.use_target_q:
                next_q_tot = self._compute_qs(inputs=next_inputs,
                                              qnet=self.qnet2,
                                              mixer=self.mixer2)

        q_targets = rewards + self.gamma * next_q_tot * (1 - dones)

        loss = torch.nn.functional.mse_loss(input=q_tot, target=q_targets)

        if self.fit_conf['auto_norm_clip']:
            norm_clip_val = self.fit_conf['auto_norm_clip_base_val'] * q_tot.shape[0]
        else:
            norm_clip_val = self.fit_conf['norm_clip_val']

        params = list(self.qnet.parameters()) + list(self.mixer.parameters())

        self.clip_and_optimize(optimizer=self.qnet_optimizer,
                               params=params,
                               loss=loss,
                               clip_val=norm_clip_val)

        fit_dict = dict()
        fit_dict['loss'] = loss.detach().cpu().numpy()

        if self.use_clipped_q:
            loss2 = torch.nn.functional.mse_loss(input=q_tot2, target=q_targets)
            params = list(self.qnet2.parameters()) + list(self.mixer2.parameters())
            self.clip_and_optimize(optimizer=self.qnet2_optimizer,
                                   params=params,
                                   loss=loss2,
                                   clip_val=norm_clip_val)

            fit_dict['loss2'] = loss2.detach().cpu().numpy()

        else:
            self.update_target_network(self.fit_conf['tau'], self.qnet, self.qnet2)
            self.update_target_network(self.fit_conf['tau'], self.mixer, self.mixer2)

        return fit_dict