"""Hidden markov model module.

This module implements logic related to hidden markov models. In particular
it deals with the type of casino model described in section 2.2 of
assignment 2.
"""

__author__ = 'Cedric Seger'

import numpy as np
import copy
import logging


class HMM():
    """Implements logic for a hidden markov model (HMM).

    Parameters
    ----------
    starting_prior : list
        List containing the starting priors for each state.
    evidence_matrix : 2D array
        List of lists specifying the categorical distributions for each table
        group.
    transition_matrix : 2D array
        List of lists representing the HMM transition table.
    prob_observe : float
        Number representing the probability of observing an outcome.
    """
    def __init__(self, starting_prior, evidence_matrix,
                 transition_matrix, prob_observe):
        self.starting_prior = starting_prior
        self.transition_matrix = transition_matrix
        self.evidence_matrix = evidence_matrix
        self.num_states = len(starting_prior)
        self.prob_observe = prob_observe
        self.state = None
        self.time_step = None
        self.actual_outcomes = []
        self.observed_outcomes = []
        self.sample_starting_state()

    def sample_starting_state(self):
        """Generates a new starting state for casino model."""
        state = np.random.multinomial(1, self.starting_prior)
        self.state = np.argmax(state)
        self.time_step = 1
        logging.info('Starting State: {}'.format(self.state))

    def sample_observation(self):
        """Samples an observation from current state."""
        observation = np.random.multinomial(
            1, self.evidence_matrix[self.state])
        # Add 1 so that observations are in range (1...6)
        return np.argmax(observation) + 1

    def transition(self):
        """Generates next state based on transition table and current state."""
        next_state = np.random.multinomial(
            1, self.transition_matrix[self.state])
        self.state = np.argmax(next_state)
        self.time_step += 1

    def run(self):
        """Convenenience method for sampling an observation and next state."""
        observation = self.sample_observation()
        self.actual_outcomes.append(observation)
        if np.random.uniform(0, 1) <= self.prob_observe:
            self.observed_outcomes.append(observation)
        else:
            # We do not get to observe outcome, represent as (-1)
            self.observed_outcomes.append(-1)
        self.transition()

    def reset(self):
        """Resets state and other information of current markov model."""
        self.state = None
        self.time_step = None
        self.actual_outcomes = []
        self.observed_outcomes = []
        self.sample_starting_state()

    def print_info(self):
        """Prints information that describes the HMM sampling sequence."""
        print('')
        print('### INFO FOR HMM ###')
        print('Time Step: {}'.format(self.time_step))
        print('Current State: {}'.format(self.state))
        print('Observed Outcomes: {}'.format(self.observed_outcomes))
        print('Actual Outcomes: {}'.format(self.actual_outcomes))

    def _semi_forward(self, observations, target_state):
        """Calculates the joint probability of past observations and a state.

        Finds the probability p(O_{1:k-1}, z_k), that is the joint probability
        of past observations from step 1 to k and the current state of node
        k, z_k.

        Parameters
        ----------
        observations : list
            List of observations from time period 1:k.
        target_state : int
            The state of node k. Can be either 0 or 1.
        """
        if len(observations) == 0:
            return self.starting_prior[target_state]
        observations.reverse()
        forward = self._forward_init(observations.pop())
        if len(observations) != 0:
            forward = self.forward(forward, observations)
        total_sum = 0
        for prev_state in range(self.num_states):
            transition_prob = self.transition_matrix[prev_state][target_state]
            total_sum += forward[prev_state] * transition_prob
        return total_sum

    def _forward_init(self, first_observation):
        forwards = np.empty(self.num_states)
        for state in range(self.num_states):
            evidence_prob = self.prob_observation(first_observation, state)
            forwards[state] = evidence_prob * self.starting_prior[state]
        return forwards

    def prob_observation(self, current_observation, current_state):
        """Calculates the probability of an observation given a state.

        Corresponds to calculating the probability: p(O_k | z_k).

        Parameters
        ----------
        current_observation : int
            Integer in range (-1,1-6). -1 implies that the observation is
            hidden.
        current_state : int
            State of latent variable z_k. Either 0 or 1.
        """
        if current_observation == -1:
            return 1 - self.prob_observe
        else:  # Observations are from 1 to 6 but index runs from 0...5
            return (
                self.prob_observe
                * self.evidence_matrix[current_state][current_observation-1])

    def forward(self, prev_forwards, observations):
        """Calculates the value of the forward table.

        The forward is the joint probability: p(O_{2:k}, z_k) for k > 1.
        The function calculates this value recursively, in polynomial time.

        Parameters
        ----------
        prev_forwards : list
            Table of forwards from a previous time step.
        observations : list
            List of integers representing observations at nodes.

        Returns:
            Table that specifies the forward values for different states
            of variable z_k.
        """
        current_observation = observations.pop()
        current_forward = np.empty(self.num_states)
        for current_state in range(self.num_states):
            evidence_prob = self.prob_observation(
                current_observation, current_state)
            total_sum = 0
            for prev_state in range(self.num_states):
                transition_prob = (
                    self.transition_matrix[prev_state][current_state])
                total_sum += prev_forwards[prev_state] * transition_prob
            current_forward[current_state] = evidence_prob * total_sum
        if (len(observations) == 0):
            return current_forward
        future_forward = self.forward(current_forward, observations)
        return future_forward

    def backward(self, observations, future_backward, target_state):
        """Calculates the value of the backward.

        The backward corresponds to finding the probability of a sequence of
        future observations given a state at time k. Formally, this can be
        written as the probability: p(O_(k+1:T)| z_k).

        Parameters
        ----------
        observations : list
            List of observations from time period K+1:T. T being the final
            time step (or node) in the sampling sequence.
        future_backward : list
            The backward value from a future state.
            Originally set to a list of ones.
        target_state : int
            The state of the node of interest.
        """
        current_backward = np.empty(self.num_states)
        future_observation = observations.pop()
        for current_state in range(self.num_states):
            total_sum = 0
            for future_state in range(self.num_states):
                transition_prob = (
                    self.transition_matrix[current_state][future_state])
                future_evidence = self.prob_observation(
                    future_observation, future_state)
                total_sum += (transition_prob
                              * future_evidence
                              * future_backward[future_state])
            current_backward[current_state] = total_sum
        if len(observations) != 0:
            return self.backward(observations, current_backward, target_state)
        # Else we are done
        return current_backward[target_state]

    def calculate_conditional(self, k, s, observations_list):
        """Calculates the conditional probability: p(x_k, z_k | S, O_{1:T}).

        Finds the conditional probability of an observation x_k and state
        z_k given the sum of all outcomes and a set of observations from each
        time step/node in the HMM.

        Parameters
        ----------
        k : int
            The node that we are interested in. A particular time step in the
            sequence model.
        s : int
            The sum of all outcomes in the sequence.
        observations_list : list
            List of integers representing observations. Can be -1 or in (1,6).

        Returns:
            Table of probabilities with entries for each
            (state, observation) pair.
        """
        num_outputs = len(self.evidence_matrix[0])
        prob_table = np.zeros([self.num_states, num_outputs])
        k = k - 1  # Make index friendly
        for state in range(self.num_states):
            for x in range(num_outputs):
                joint_prob = self._calculate_joint(
                    x+1, state, k, s, observations_list)
                prob_table[state, x] = joint_prob
        prob_table = prob_table / np.sum(prob_table)
        return prob_table

    def _calculate_joint(self, x_k, z_k, k, s, observations_list):
        """Calculates the joint probability: p(X_k=x, Z_k=t, S=s, O_1:T).

        Parameters
        ----------
        x_k : int
            An observation related to state z_k.
        z_k : int
            The state of node k. Either 0 or 1.
        k : int
            The current timestep. Node number k.
        s : int
            The sum associated with all the T observations from the HMM.
        observations_list : list
            List of observations.

        Returns:
            Joint probability.
        """
        observations_list_cp = copy.deepcopy(observations_list)
        past_observations = observations_list_cp[:k]
        future_observations = observations_list_cp[k+1:]
        current_observation = observations_list_cp[k]
        # Case that cannot happen. I.e. O_k = 5 and X_k = 1
        if (current_observation != -1) and (current_observation != x_k):
            return 0
        forward_value = self._semi_forward(past_observations, z_k)
        prob_observation_k = self.prob_observation(current_observation, z_k)
        backward_value = 1
        if len(future_observations) != 0:
            backward_value = self.backward(
                future_observations, future_backward=[1, 1], target_state=z_k)
        z_list = [-1 for i in range(len(observations_list_cp))]
        z_list[k] = z_k
        observations_list_cp[k] = x_k
        prob_sum = self.prob_sum(s, observations_list_cp, z_list)
        return forward_value * prob_observation_k * backward_value * prob_sum

    def _intermediate_prob(self, current_state, current_node,
                           target_sum, prev_tablesum, transition_matrix,
                           evidence_matrix):
        """Calculates probability of being in a state and observing a sum."""
        total_prob_sum = 0
        for prev_state in range(self.num_states):
            for prev_sum in range(current_node-1, (current_node-1)*6+1):
                # Subtract 1 to get into proper index
                sum_index = prev_sum - (current_node - 1)
                target_sum_index = target_sum - prev_sum - 1
                # Check if evidence is possible... a dice only has 6 sides
                if (target_sum_index < 0) or (target_sum_index > 5):
                    total_prob_sum += 0
                else:
                    prob_previous_sum = prev_tablesum[prev_state][sum_index]
                    transition_prob = (
                        transition_matrix[prev_state][current_state])
                    evidence_prob = (
                        evidence_matrix[current_state][target_sum_index])
                    total_prob_sum += (
                        prob_previous_sum * transition_prob * evidence_prob)
        return total_prob_sum

    def find_prob_sum(self, current_node, prev_tablesum,
                      observations_list, z_list):
        """Runs the recursive algorithm to the table of probabilities.

        Calculates and returns the table of probabilities specifying
        probabilities of (state, sum) pairs.
        """
        observation = observations_list.pop()
        z_k = z_list.pop()
        transition_matrix, evidence_matrix = self._setup_matrices(
            z_k, observation)
        current_tablesum = np.empty(
            [self.num_states, current_node*6 - current_node + 1])
        for state in range(self.num_states):
            for target_sum in range(current_node, current_node*6+1):
                target_sum_index = target_sum - current_node
                prob = self._intermediate_prob(
                    state, current_node, target_sum,
                    prev_tablesum, transition_matrix, evidence_matrix)
                current_tablesum[state][target_sum_index] = prob

        if (len(observations_list) == 0):  # We are done
            return current_tablesum
        future_tablesum = self.find_prob_sum(
            current_node+1, current_tablesum, observations_list, z_list)
        return future_tablesum

    def _setup_matrices(self, z_k, observation):
        """Sets up and adjusts transition and evidence matrices.

        If an observation or state (z_k) is known, then this affects the
        transition and evidence probability tables of the HMM respectively.
        """
        transition_matrix = copy.deepcopy(self.transition_matrix)
        evidence_matrix = copy.deepcopy(self.evidence_matrix)
        if (z_k != -1):
            given_state = z_k
            other_state = (given_state + 1) % 2
            for state in range(self.num_states):
                transition_matrix[state][given_state] = 1
                transition_matrix[state][other_state] = 0
        if (observation != -1):
            evidences = [0]*6
            evidences[observation-1] = 1
            for state in range(self.num_states):
                evidence_matrix[state] = evidences
        return transition_matrix, evidence_matrix

    def prob_sum(self, observed_sum, observations_list, z_list):
        """Calculates the probability of a sum given states and observations.

        Finds the probability: p(S | O_{1:k}, z_list). That is, the probability
        of a sum given a set of known observations and a set of known states.

        Parameters
        ----------
        observed_sum : int
            The sum of interest.
        observations_list : list
            List of observations associated with each state in z_list.
            The observations are integers and can be in the range (-1,6). A
            -1 indicates that the observation is hidden.
        z_list : list
            Contains integers representing HMM states. If -1, then it means
            state is not known. Integers 0 and 1 represent actual known states.

        Returns:
            Probability of the sum.
        """
        num_nodes = len(observations_list)
        if (observed_sum < num_nodes) or (observed_sum > num_nodes*6):
            return 0

        observations_list.reverse()
        z_list.reverse()
        observation_k = observations_list.pop()
        z_k = z_list.pop()
        # Initialize table for algorithm
        init_tablesum = copy.deepcopy(self.evidence_matrix)
        if (observation_k != -1):
            evidences = [0]*6
            evidences[observation_k-1] = 1
            for state in range(self.num_states):
                init_tablesum[state] = evidences
        if (z_k != -1):
            other_state = (z_k + 1) % 2
            init_tablesum[other_state] = [0]*6
        else:
            for state in range(self.num_states):
                init_tablesum[state] = (
                    [x * self.starting_prior[state] for x in init_tablesum[state]])

        if len(observations_list) == 0:
            tablesum = init_tablesum
        else:
            current_node = 2
            tablesum = self.find_prob_sum(
                current_node, init_tablesum, observations_list, z_list)
        sum_index = observed_sum - num_nodes
        tablesum = np.sum(tablesum, axis=0)
        prob = tablesum[sum_index]
        return prob

    def sample_z_k(self, z_list, observations_list, observed_sum):
        """Samples a state of the HMM given some information.

        Formally, this function implements logic that samples a state 'z_k'
        from either the distribution p(z_k | S, O_{1:k}) or from
        p(z_{k-1} |z_k, S, O_{1:k}).

        Parameters
        ----------
        z_list : list
            Contains integers representing HMM states. If -1, then it means
            state is not known. Integers 0 and 1 represent actual known states.
        observations_list : list
            List of observations associated with each state in z_list.
        observed_sum : int
            The sum of all observations in the observations_list.

        Returns:
            z_list: Updated z_list containing a new state for the last unknown
            state in the list prior to calling the function.
            condit_prob: np array containing the probabilities associated with
            the sampled state.
        """
        # Find the last unknown state in z_list.
        k = -1
        for i in range(1, len(z_list)+1):
            if z_list[-i] == -1:
                k = -i
                break

        probability_table = np.zeros(self.num_states)
        for state in range(self.num_states):
            observations_list_cp = copy.deepcopy(observations_list)
            current_observation = observations_list_cp[k]
            past_observations = observations_list_cp[:k]
            if k + 1 == 0:
                future_observations = []
            else:
                future_observations = observations_list_cp[k+1:]
            # Find probability of past observations and states
            z_list[k] = state
            prob_sum = self.prob_sum(
                observed_sum, observations_list_cp, copy.deepcopy(z_list))
            forward_value = self._semi_forward(past_observations, state)
            prob_observation_k = self.prob_observation(
                current_observation, state)
            alpha_k = forward_value * prob_observation_k
            # Find probability of future obserations and states
            future_prob = 1
            prev_state = state
            if len(future_observations) != 0:
                for i in range(len(future_observations)):
                    future_obs = future_observations[i]
                    future_state = z_list[k+i+1]
                    prob_observation = self.prob_observation(
                        future_obs, future_state)
                    transition_prob = (
                        self.transition_matrix[prev_state][future_state])
                    future_prob *= prob_observation * transition_prob
                    prev_state = future_state
            probability_table[state] = alpha_k * prob_sum * future_prob
        condit_prob = probability_table / np.sum(probability_table)
        z_k = np.argmax(condit_prob)
        z_list[k] = z_k
        return z_list, condit_prob

    def sample_states(self, observations_list, observed_sum):
        """Samples the most likely sequence of states given observations.

        Finds the set of states that are most likely given a set of
        observations drawn from the hidden markov model. Formally this can be
        regarded as sampling from p(Z_1, Z_2,..., Z_k | S, O_{1:k}), where
        S is the observed sum and O_{1:k} are the observations.

        Parameters
        ----------
        observations_list : list
            List containing the observations from each state
        observed_sum : int
            The sum of all observations

        Returns:
            states_list: List containing the most likely states
            prob_list: List containing the probabilities associated
            with each state
        """
        states_list = [-1 for x in observations_list]
        prob_list = []
        # Samples states in reverse order. From k to 1.
        for i in range(len(states_list)):
            states_list, prob_table = self.sample_z_k(
                states_list, observations_list, observed_sum)
            prob_list.append(prob_table)
        prob_list.reverse()
        return states_list, prob_list
