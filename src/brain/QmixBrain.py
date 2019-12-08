import torch
import warnings

from src.config.ConfigBase import ConfigBase
from src.brain.Brainbase import BrainBase
from src.util.train_util import dn


class QmixBrainConfig(ConfigBase):
    def __init__(self, name='qmixbrain', brain_conf=None, fit_conf=None):
        super(QmixBrainConfig, self).__init__(name=name, brain=brain_conf, fit=fit_conf)

        self.brain = {
            'optimizer': 'lookahead',
            'lr': 1e-5,
            'gamma': 1.0,
            'eps': 0.5,
            'eps_gamma': 0.995,
            'eps_min': 0.01,
            'use_double_q': True,
            'use_clipped_q': True,
            'mixer_use_hidden': True,
            'use_noisy_q': True
        }

        self.fit = {
            'tau': 0.1,
            'auto_norm_clip': True,
            'auto_norm_clip_base_val': 0.1,
            'norm_clip_val': 1.0
        }


class QmixBrain(BrainBase):
    def __init__(self, conf, qnet, mixer, qnet2, mixer2):
        super(QmixBrain, self).__init__()
        self.conf = conf
        self.brain_conf = conf.brain
        self.fit_conf = conf.fit
        self.mixer_use_hidden = self.brain_conf['mixer_use_hidden']

        self.use_double_q = self.brain_conf['use_double_q']
        self.use_clipped_q = self.brain_conf['use_clipped_q']

        self.gamma = self.brain_conf['gamma']
        self.register_buffer('eps', torch.ones(1, ) * self.brain_conf['eps'])
        self.register_buffer('eps_min', torch.ones(1, ) * self.brain_conf['eps_min'])
        self.eps_gamma = self.brain_conf['eps_gamma']

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

        if self.use_target_q:
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

    def get_action(self, use_q1=True, **inputs):
        inputs["eps"] = self.eps

        if use_q1:
            qnet = self.qnet
        else:
            qnet = self.qnet2

        nn_actions, info_dict = qnet.get_action(**inputs)

        return nn_actions, info_dict

    def _compute_qs(self, inputs: dict, qnet, mixer, actions=None):
        q_dict = qnet.compute_qs(**inputs)
        qs = q_dict['qs']
        if actions is None:
            qs, _ = qs.max(dim=1)
        else:
            qs = qs.gather(-1, actions.unsqueeze(-1).long()).squeeze(dim=-1)
        if self.mixer_use_hidden:
            q_tot = mixer(inputs['curr_graph'], q_dict['hidden_feat'], qs)
        else:
            q_tot = mixer(inputs['curr_graph'], inputs['curr_feature'], qs)
        return q_tot

    def _compute_double_qs(self, inputs: dict, target_qnet, target_mixer, action_qnet):
        action_q_dict = action_qnet.compute_qs(**inputs)
        action_q = action_q_dict['qs']
        actions = action_q.argmax(dim=1)

        target_q_dict = target_qnet.compute_qs(**inputs)
        target_q = target_q_dict['qs']
        target_q = target_q.gather(-1, actions.unsqueeze(-1).long()).squeeze(dim=-1)
        if self.mixer_use_hidden:
            target_q_tot = target_mixer(inputs['curr_graph'], target_q_dict['hidden_feat'], target_q)
        else:
            target_q_tot = target_mixer(inputs['curr_graph'], inputs['curr_feature'], target_q)
        return target_q_tot

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
        fit_dict['loss'] = dn(loss)

        if self.use_clipped_q:
            loss2 = torch.nn.functional.mse_loss(input=q_tot2, target=q_targets)
            params = list(self.qnet2.parameters()) + list(self.mixer2.parameters())
            self.clip_and_optimize(optimizer=self.qnet2_optimizer,
                                   params=params,
                                   loss=loss2,
                                   clip_val=norm_clip_val)

            fit_dict['loss2'] = dn(loss2)

        else:
            self.update_target_network(self.fit_conf['tau'], self.qnet, self.qnet2)
            self.update_target_network(self.fit_conf['tau'], self.mixer, self.mixer2)

        return fit_dict
