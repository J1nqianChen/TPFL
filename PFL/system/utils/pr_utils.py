import numpy as np


class WorkerSampler:
    def __init__(self, method, participation_prob, cycle=None, trans_prob=None):
        self.participation_prob = participation_prob

        self.method = method

        if self.method == 'cyclic':
            if cycle is None:
                raise RuntimeError
            self.cycle = cycle
            self.active_rounds = max(1, int(np.round(self.cycle * self.participation_prob)))
            self.inactive_rounds = max(1, self.cycle - self.active_rounds)
            self.currently_active = False
            self.rounds_to_switch = np.random.randint(self.inactive_rounds)

        elif self.method == 'markov':
            if trans_prob is None:
                raise RuntimeError
            self.transition_to_active_prob = trans_prob
            self.transition_to_inactive_prob = self.transition_to_active_prob * (1 / self.participation_prob - 1)

            if self.transition_to_inactive_prob > 1:
                self.transition_to_active_prob /= self.transition_to_inactive_prob
                self.transition_to_inactive_prob = 1

            self.currently_active = (np.random.binomial(1, self.participation_prob) == 1)

    def cyclic_update(self):
        if self.rounds_to_switch == 0:
            self.currently_active = not self.currently_active
            if self.currently_active:
                self.rounds_to_switch = self.active_rounds
            else:
                self.rounds_to_switch = self.inactive_rounds
        self.rounds_to_switch -= 1

    def markov_update(self):
        if self.currently_active:
            if np.random.binomial(1, self.transition_to_inactive_prob) == 1:
                self.currently_active = False
        else:
            if np.random.binomial(1, self.transition_to_active_prob) == 1:
                self.currently_active = True

    def sample(self):
        if self.method == 'bernoulli':
            return np.random.binomial(1, self.participation_prob) == 1
        elif self.method == 'cyclic':
            self.cyclic_update()
            return self.currently_active
        elif self.method == 'markov':
            self.markov_update()
            return self.currently_active


def get_pr_table(method, num_clients, total_round, pr, trans_prob=None, cycle=None):
    list_client_pr = []
    for i in range(num_clients):
        list_action = []
        sample_instance = WorkerSampler(method=method, participation_prob=pr, cycle=cycle, trans_prob=trans_prob)
        for j in range(total_round):
            rs = sample_instance.sample()
            list_action.append(rs)
        list_client_pr.append(np.array(list_action))
    full_pr_tab = np.stack(list_client_pr, axis=0)
    return full_pr_tab