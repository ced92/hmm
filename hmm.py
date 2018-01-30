
import numpy as np
import copy
import logging
logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')

class HMM():
    """ Class Representing HMM for Casino model
    Args:
        starting_prior: List of probabilities over each state (eg. [1/2, 1/2])
        dice_dist1: List of prob. for each outcome of dice in table group 1 (eg. [1/12]*4 + [2/6]*2)
        dice_dist1: List of prob. for each outcome of dice in table group 2
        transition_dist: list of probabilities that represents HMM transitioning table
        prob_observe: Prob. of observing outcome at time_step K
    """
    def __init__(self, starting_prior, dice_dist1, dice_dist2, transition_matrix, prob_observe):
        self.starting_prior = starting_prior
        self.dice_dist1 = dice_dist1
        self.dice_dist2 = dice_dist2
        self.transition_matrix = transition_matrix
        self.evidence_matrix = [dice_dist1, dice_dist2]
        self.num_states = len(starting_prior)

        self.prob_observe = prob_observe
        self.state = None
        self.time_step = None
        self.actual_outcomes = []
        self.observed_outcomes = [] # We do not get to observe all outcomes...

        self.sample_starting_state() # INIT

    def sample_starting_state(self):
        """Generates starting state of model"""
        state = np.random.multinomial(1, self.starting_prior)
        self.state = np.argmax(state)
        self.time_step = 1
        logging.info('Starting State: {}'.format(self.state))

    def sample_observation(self):
        """Samples an observation from current state"""
        observation = np.random.multinomial(1, self.evidence_matrix[self.state])
        return np.argmax(observation) + 1 # Add 1 so that observations starts from 1...6

    def transition(self):
        """Generates next state based on transition table"""
        next_state = np.random.multinomial(1, self.transition_matrix[self.state])
        self.state = np.argmax(next_state)
        self.time_step += 1


    def run(self):
        """Runs one iteration of main loop (Sample -> Transition + Store Information)"""
        # Sample
        observation = self.sample_observation()
        self.actual_outcomes.append(observation)

        # Store information
        if np.random.uniform(0,1) <= self.prob_observe:
            # We get to "observe" the observation
            self.observed_outcomes.append(observation)
        else:
            # We do not get to observe outcome, represent as (-1)
            self.observed_outcomes.append(-1)

        # Next State
        self.transition()

    def reset(self):
        self.state = None
        self.time_step = None
        self.actual_outcomes = []
        self.observed_outcomes = []
        self.sample_starting_state() # INIT

    def print_info(self):
        print('')
        print('### INFO FOR HMM ###')
        print('Time Step: {}'.format(self.time_step))
        print('Current State: {}'.format(self.state))
        print('Observed Outcomes: {}'.format(self.observed_outcomes))
        print('Actual Outcomes: {}'.format(self.actual_outcomes))


    #### METHODS FOR Algorithms ####

    def semi_forward(self, observations, target_state):
        """ Calculates: prob(O_(1:K-1), Z_K = h)

        Variables:
            target_state: An integer representing the current state of hidden variable Z at time K. {0,1}
            observations: List of observations from time period 1:K-1. K being the current time step.
        """
        # Only the case when the node (Z_k) corresponds to the first node i.e. Z_1
        # Then just return prob(Z_k) since we have no observations before Z_1
        if len(observations) == 0:
            return self.starting_prior[target_state]

        # Else, we must have at least one observation. This implies k >= 2.
        observations.reverse() # Reverse so that we can pop from end of list corresponding to first time step
        # print('Reversed observations: {}'.format(observations))
        forward = self.forward_init(observations.pop()) # Get initial starting condition for forward
        # print('Init forward value: {}'.format(forward))

        if len(observations) != 0:
            # print('We are recursing...')
            forward = self.forward(forward, observations) # Returns array of length = num_hidden_states

        # print('Final Forward values: {}'.format(forward))
        total_sum = 0
        for prev_state in range(self.num_states): # Sum over all previous states
            transition_prob = self.transition_matrix[prev_state][target_state]
            total_sum += forward[prev_state] * transition_prob

        return total_sum


    def forward_init(self, first_observation):
        forwards = np.empty(self.num_states) # Array to hold values for each state
        for state in range(self.num_states): # Loop through each state
            evidence_prob = self.prob_observation(first_observation, state)
            forwards[state] = evidence_prob * self.starting_prior[state]

        return forwards

    def emmission(self, target_observation, target_evidence, target_state):
        """ Calculates prob: p(O_k = o, X_k = x | Z_k = t)

        """
        # Case that cannot happen. I.e. O_k = 5 and X_k = 1 are mutually exclusive events
        if (target_observation != -1) and (target_observation != target_evidence):
            print('Emission O_K != Evidence X_K -> All prob will be zero')
            return 0
        prob_evidence = self.evidence_matrix[target_state][target_evidence-1]
        if target_observation == -1: # Unobserved outcome
            return prob_evidence * (1 - self.prob_observe)
        else: # Outcome is observed
            return prob_evidence * self.prob_observe

    def prob_observation(self, current_observation, current_state):
        """ Calculates P(O_k | Z_k)

        Variables:
            current_observation: Integer in range (1,6) representing observation from hidden state
            current_state: Integer in range (0,1) representing the state of latent variable
        """
        # Int: -1 represents an "unobserved" observation
        if current_observation == -1:
            return 1 - self.prob_observe
        else: # Observations are from 1 to 6 but index runs from 0...5
            return self.prob_observe * self.evidence_matrix[current_state][current_observation-1]

    def forward(self, prev_forwards, observations): #TODO: Change this so we dont condition on particular Z_K
        """  Calculates the forward f_k(h) := p(O_(2:K), Z_K = h) for K > 1

        Variables:
            current_state: An integer representing the current state of hidden variable Z at time K. {0,1}
            observations: List of observations from time period 1:K. K being the current time.
        """

        current_observation = observations.pop() # Removes and returns last element of list
        # print('Popped observation: {}'.format(current_observation))
        # print('Remaining observations: {}'.format(observations))
        # print('Upon entering... prev_forwards is : {}'.format(prev_forwards))
        current_forward = np.empty(self.num_states) # To hold current forward values

        for current_state in range(self.num_states): # Loop through each current state
            evidence_prob = self.prob_observation(current_observation, current_state)
            # print('prob of {} in state {} is: {}'.format(current_observation,current_state,evidence_prob))

            total_sum = 0
            for prev_state in range(self.num_states): # Sum over all previous states
                transition_prob = self.transition_matrix[prev_state][current_state]
                # print('prob of {} -> {} is: {}'.format(prev_state,current_state,transition_prob))
                total_sum += prev_forwards[prev_state] * transition_prob

            # print('Total sum is : {}'.format(total_sum))
            # Update current_forward value
            current_forward[current_state] = evidence_prob * total_sum

        # print('Current_forward is : {}'.format(current_forward))

        # We have forwarded through all time steps until K-1
        if (len(observations) == 0):
            return current_forward

        future_forward = self.forward(current_forward, observations) # Recursive call moving forward

        # Return to top of calling stack
        return future_forward


    def backward(self, observations, future_backward, target_state):
        """  Calculates the backward f_k(Z) := p(O_(K+1:T)| Z_K)

        Variables:
            future_backward: The backward value from a future state. Originally set to vector of ones.
            observations: List of observations from time period K+1:T. K being the current time step and T the final.
        """
        current_backward = np.empty(self.num_states)
        future_observation = observations.pop()

        for current_state in range(self.num_states):
            total_sum = 0
            for future_state in range(self.num_states):
                transition_prob = self.transition_matrix[current_state][future_state]
                future_evidence = self.prob_observation(future_observation, future_state)
                total_sum += transition_prob * future_evidence * future_backward[future_state]

            # Update current_backward value
            current_backward[current_state] = total_sum

        # If there are observations left, keep recursing...
        if len(observations) != 0:
            return self.backward(observations, current_backward, target_state)

        # Else we are done, lets return appropriate value to caller...
        return current_backward[target_state]

    def find_prob_sum(self, observed_sum, final_node, observations_list, z_k):
        """ Calculates P(Sum_k = observed_sum) I.e. that the sum of k dices is observed_sum

        Variables:
            observed_sum: Sum that we are interested in
            final_node: Integer value representing how many time steps to consider
            observations_list: List of observations from time period 1:T.
            z_k: Triplet in form (k, z_k, x_k). k specifying time step, z_k state of node z_k and x_k the outcome at time step k.
        """
        # Basic checks.
        if (final_node > len(observations_list)):
            logging.warning('Error in input. Value of final_node: {0} > total number of nodes: {1}'.format(final_node, len(observations_list)))
            return
        if (z_k[0] > len(observations_list)) or (z_k[0] < 1):
            logging.warning('Value of k in z_k triple must be between 1 and T')
            return
        if (z_k[1] > 1) or (z_k[1] < 0):
            logging.warning('Value of z_k was : {} but must be between 0 and 1'.format(z_k[1]))
            return

        observations_list.reverse()
        observation = observations_list.pop() # Pop last element which is now first from reverse
        init_tablesum = copy.deepcopy(self.evidence_matrix) # Make deepcopy

        # Adjust init table-sum accordingly. Before main recursion begins.
        if (z_k[0] == 1): # If it is specified that this node is given info z_k
            for state in range(self.num_states):
                evidences = [0]*6
                evidences[z_k[2]-1] = 1
                init_tablesum[state] = evidences
                if(state != z_k[1]):
                    init_tablesum[state] = [0]*6
#             print('Adjusted evidence: {}'.format(evidences))
        else:
            if (observation != -1): # I.e. O_1 = 2 (observed outcome)
                for state in range(self.num_states):
                    evidences = [0]*6
                    evidences[observation-1] = 1
                    init_tablesum[state] = evidences
#                 print('Adjusted evidence: {}'.format(evidences))

            for state in range(self.num_states):
                init_tablesum[state] = [x * self.starting_prior[state] for x in init_tablesum[state]]

        if final_node == 1:
            tablesum = init_tablesum
        else:
            # Else run algorithm...
            current_node = 2
            tablesum = self.prob_sum(observed_sum, current_node, final_node, init_tablesum, observations_list, z_k)

        if (observed_sum < final_node) or (observed_sum > final_node*6):
            return 0

        prob_sum = 0
        sum_index = observed_sum - final_node # Gives appropriate index
        for state in range(self.num_states):
            prob_sum += tablesum[state][sum_index]
        return prob_sum

    def prob_sum(self, observed_sum, current_node, final_node, prev_tablesum, observations_list, z_k):
        observation = observations_list.pop()
        transition_matrix = copy.deepcopy(self.transition_matrix) # Make sure to deep-copy!
        evidence_matrix = copy.deepcopy(self.evidence_matrix)

        # If we are given some information, adjust probabilities accordingly. Then continue as normal.
        if (current_node == z_k[0]): # We are given this node...
            given_state = z_k[1]
            other_state = (given_state + 1) % 2
            evidences = [0]*6
            evidences[z_k[2]-1] = 1
            for state in range(self.num_states):
                transition_matrix[state][given_state] = 1
                transition_matrix[state][other_state] = 0
                evidence_matrix[state] = evidences
        elif(observation != -1): # We are given an observation
            evidences = [0]*6
            evidences[observation-1] = 1
            for state in range(self.num_states):
                evidence_matrix[state] = evidences

        current_tablesum = np.empty([self.num_states, current_node*6 - current_node + 1])
        for state in range(self.num_states):
            for target_sum in range(current_node, current_node*6+1):
                target_sum_index = target_sum - current_node
                current_tablesum[state][target_sum_index] = self._intermediate_prob(state,
                                                                                    current_node,
                                                                                    target_sum,
                                                                                    prev_tablesum,
                                                                                    transition_matrix,
                                                                                    evidence_matrix)
        # We are done.
        if (current_node == final_node):
            return current_tablesum

        # Otherwise recurse forward
        future_tablesum = self.prob_sum(observed_sum, current_node+1, final_node, current_tablesum, observations_list, z_k)
        return future_tablesum

    def _intermediate_prob(self, current_state, current_node, target_sum ,prev_tablesum, transition_matrix, evidence_matrix):
        # (state, current_node, target_sum, prev_tablesum, transition_matrix, evidence_matrix)
        total_prob_sum = 0
        for prev_state in range(self.num_states):
            # Loop over all possible sums in previous state
            for prev_sum in range(current_node-1, (current_node-1)*6+1):
                # Subtract 1 to get into proper index
                sum_index = prev_sum - (current_node-1)
#                 print('Prev sum index: {}'.format(sum_index))
                target_sum_index = target_sum - prev_sum - 1 # -1 for indexing...
                # Check if evidence is possible... a dice only has 6 sides
                if (target_sum_index < 0) or (target_sum_index > 5):
                    total_prob_sum += 0
                else:
#                     print('Target sum index: {}'.format(target_sum_index))
#                     print('Target sum --- : {}'.format(target_sum_index + 1))
#                     print('Prev sum --- : {}'.format(prev_sum))
                    prob_previous_sum = prev_tablesum[prev_state][sum_index]
                    transition_prob = transition_matrix[prev_state][current_state]
                    evidence_prob = evidence_matrix[current_state][target_sum_index]
                    total_prob_sum += prob_previous_sum * transition_prob * evidence_prob
        return total_prob_sum


    def calculate_conditional(self, k, s, observations_list):
        """ Calculates the cond. prob: p(X_k = x, Z_k = t | S_N, O_1:T)"""
#         # Basic Checks
#         if (z_k > 1) or (z_k < 0):
#             print('z_k must be between 0 and 1 but was found to be {}'.format(z_k))
#             return

        prob_table = np.zeros([self.num_states, 6])

        # 1 - Find joint we are interested in
#         target_value = self._calculate_joint(x_k,z_k,k,s,observations_list)
        target_value = 0

        # 2 - Find sum over all combinations of X_k, Z_k (i.e. the marginalizer)
        total_sum = 0
        for state in range(self.num_states):
            for x in range(1,7):
                result = self._calculate_joint(x,state,k,s,observations_list)
                logging.debug('Result for state {}, dice {} is : {}'.format(state,x,result))
                prob_table[state, x-1] = result
                total_sum += result
#                 if (x == x_k) and (state == z_k):
#                     target_value = result

        # Normalize probabilities
        if total_sum == 0:
            total_sum = 1 # Avoids dividing by zero.

        prob_table = prob_table / total_sum
        return prob_table

        # 3 - Calculate cond. probability (marginalize)
#         print('Final target value: {}'.format(target_value))
#         print('Final Total sum: {}'.format(total_sum))
#         if (target_value == 0): # Avoid 0 / 0
#             return 0
#         return target_value / total_sum

    def _calculate_joint(self, x_k, z_k, k, s, observations_list):
        """ Calculates joint prob: p(X_k=x, Z_k=t, S=s, O_1:T)

            Variables:
            k: Integer representing a specific time step. Must be between 1 and T.
            z_k: Integer representing state {0,1}

        """
        observations_list_cp = copy.deepcopy(observations_list) # Force copy. Python does not pass by value?
#         print('K is: {}'.format(k))
#         print('Observations list is: {}'.format(observations_list))
        # 1 - Partition observations according to z_k
        past_observations = observations_list_cp[:k-1]
        future_observations = observations_list_cp[k:]
        current_observation = observations_list_cp[k-1]
#         print('Past obs: {}'.format(past_observations))
#         print('Future obs: {}'.format(future_observations))
#         print('Current obs: {}'.format(current_observation))

        # 2 - Run all algorithms seperately
        logging.debug('Past obs: {}'.format(past_observations))
        logging.debug('Target_state = {}'.format(z_k))
        forward_value = self.semi_forward(observations=past_observations, target_state=z_k)
        backward_value = 1
        if len(future_observations) != 0:
            backward_value = self.backward(observations=future_observations, future_backward=[1,1], target_state=z_k)

#         print(current_observation)
        logging.debug('Target_observation = {}'.format(current_observation))
        logging.debug('Target_evidence = {}'.format(x_k))

        emission_value = self.emmission(target_observation=current_observation, target_evidence=x_k, target_state=z_k)
#         target_observation, target_evidence, target_state
#         print(observations_list)
        sum_value = self.find_prob_sum(observed_sum=s, final_node=len(observations_list),
                                       observations_list=copy.deepcopy(observations_list), z_k=(k, z_k, x_k))

        logging.info('Sum_value is : {}'.format(sum_value))
        logging.info('Emission Value is: {}'.format(emission_value))
        logging.info('Backward value is : {}'.format(backward_value))
        logging.info('Forward Value is : {}'.format(forward_value))
        # 3 - Multiply all values and return
        return forward_value * emission_value * backward_value * sum_value

    def prob_sum_alt(self, current_node, prev_tablesum, observations_list, z_list):
        transition_matrix = copy.deepcopy(self.transition_matrix)
        evidence_matrix = copy.deepcopy(self.evidence_matrix)

        observation = observations_list.pop()
        z_k = z_list.pop()
        logging.debug("Remaining Z_List : {}".format(z_list))
        logging.debug("Remaining Obs: {}".format(observation))
        # Perform two Checks
        if (z_k != -1): # Adjust transition probabilities
            given_state = z_k
            other_state = (given_state + 1) % 2
            logging.debug("Given state: {}".format(z_k))
            logging.debug("Other state: {}".format(other_state))

            for state in range(self.num_states):
                transition_matrix[state][given_state] = 1
                transition_matrix[state][other_state] = 0
        if (observation != -1): # Adjust evidences
            evidences = [0]*6
            evidences[observation-1] = 1
            for state in range(self.num_states):
                evidence_matrix[state] = evidences
        logging.debug("Observation: {}".format(observation))
        logging.debug("Evidence matrix: {}".format(evidence_matrix))

        # Calculate sums
        current_tablesum = np.empty([self.num_states, current_node*6 - current_node + 1])
        for state in range(self.num_states):
            for target_sum in range(current_node, current_node*6+1):
                target_sum_index = target_sum - current_node
                prob = self._intermediate_prob(state, current_node, target_sum, prev_tablesum, transition_matrix, evidence_matrix)
                current_tablesum[state][target_sum_index] = prob

        logging.debug("Current table sum: {}".format(current_tablesum))
        # We are done.
        if (len(observations_list) == 0):
            logging.debug("######### DONE #######")
            return current_tablesum

        # Otherwise recurse forward
        future_tablesum = self.prob_sum_alt(current_node+1, current_tablesum, observations_list, z_list)
        return future_tablesum

    def find_sum(self, observed_sum, observations_list, z_list):
        """ Calculates P(Sum_k | O_1:k, Zks)

        Variables:
            observed_sum: Sum that we are interested in
            final_node: Integer value representing how many time steps to consider
            observations_list: List of observations from time period 1:K.
            z_list: List of states Z_k
            z_k: Triplet in form (k, z_k, x_k). k specifying time step, z_k state of node z_k and x_k the outcome at time step k.
        """
        # # Basic checks.
        # if (final_node > len(observations_list)):
        #     print('Error in input. Value of final_node: {0} > total number of nodes: {1}'.format(final_node, len(observations_list)))
        #     return
        # if (z_k[0] > len(observations_list)) or (z_k[0] < 1):
        #     print('Value of k in z_k triple must be between 1 and T')
        #     return
        # if (z_k[1] > 1) or (z_k[1] < 0):
        #     print('Value of z_k was : {} but must be between 0 and 1'.format(z_k[1]))
        #     return

        num_nodes = len(observations_list)
        if (observed_sum < num_nodes) or (observed_sum > num_nodes*6):
            return 0

        observations_list.reverse()
        z_list.reverse()

        observation_k = observations_list.pop() # Pop last element which is now first from reverse
        z_k = z_list.pop()

        # Initialize algorithm
        init_tablesum = copy.deepcopy(self.evidence_matrix) # Make deepcopy
        logging.debug('#'*20)
        logging.debug("Init Tablesum: {}".format(init_tablesum))
        if (observation_k != -1): # O_k = {1,2,...,6}
            for state in range(self.num_states):
                evidences = [0]*6
                evidences[observation_k-1] = 1
                init_tablesum[state] = evidences
        if (z_k != -1): # z_k = {0,1}
            other_state = (z_k + 1) % 2
            init_tablesum[other_state] = [0]*6
        else: # z_k = ?
            for state in range(self.num_states): # Include transition probabilities
                init_tablesum[state] = [x * self.starting_prior[state] for x in init_tablesum[state]]

        if len(observations_list) == 0:
            tablesum = init_tablesum
        else:
            current_node = 2
            logging.debug("Tablesum before entering sum_alt_table: {}".format(init_tablesum))
            tablesum = self.prob_sum_alt(current_node, init_tablesum, observations_list, z_list)

        sum_index = observed_sum - num_nodes
        # prob_table = np.zeros(self.num_states)
        logging.debug("Table sum shape {} and {}".format(tablesum.shape, tablesum))
        logging.debug("Sum_index is: {}".format(sum_index))
        logging.debug('#'*20)
        # prob_table[0] = tablesum[0][sum_index]
        # prob_table[1] = tablesum[1][sum_index]

        # Sum up probs (since if z_k =1 was given, this will be reflected in transitions and sum will have no effect)
        # If not given, then summing is correct as well.
        tablesum = np.sum(tablesum, axis=0)
        prob = tablesum[sum_index]

        return prob
        # Returns prob(S = 18 | Z = 0) and prob(S=18 | Z=1)

    def sample_z_k(self, z_list, observations_list, observed_sum):
        """Samples from distribution: p(z_k | S, O_1:k)
        args:
            z_k: Tuple (k, value) representing state of z_k ## change to dict?
            observations_list: List of observations 1:K
        """
        k = -1
        for i in range(1, len(z_list)+1):
            logging.debug(-i)
            if z_list[-i] == -1:
                logging.debug("Success: {}".format(i))
                k = -i
                break

        probability_table = np.zeros(self.num_states)
        for state in range(self.num_states):
            observations_list_cp = copy.deepcopy(observations_list)
            ## Partition observation List into two depending on k
            current_observation = observations_list_cp[k]
            past_observations = observations_list_cp[:k]
            if k + 1 == 0:
                future_observations = []
            else:
                future_observations = observations_list_cp[k+1:]

            # Calculate prob. sum
            z_list[k] = state
            logging.debug('K is: {}'.format(k))
            logging.debug("Z_list is: {}".format(z_list))
            prob_sum = self.find_sum(observed_sum, observations_list_cp, copy.deepcopy(z_list))

            # Calculate alpha_z_k
            forward_value = self.semi_forward(past_observations, state) # check what it returns...
            prob_observation_k = self.prob_observation(current_observation, state)
            alpha_k = forward_value * prob_observation_k

            # Calculate futures (if there are any)
            future_prob = 1
            prev_state = state
            if len(future_observations) != 0:
                for i in range(len(future_observations)):
                    future_obs = future_observations[i]
                    future_state = z_list[k+i+1]
                    prob_observation = self.prob_observation(future_obs, future_state)
                    transition_prob = self.transition_matrix[prev_state][future_state]

                    future_prob *= prob_observation * transition_prob
                    prev_state = future_state

            logging.debug("alpha_k : {}".format(alpha_k))
            logging.debug("prob_sum: {}".format(prob_sum))
            logging.debug("future_prob: {}".format(future_prob))
            probability_table[state] = alpha_k * prob_sum * future_prob

        # Normalize
        logging.debug("Un-normalized prob table: {}".format(probability_table))
        condit_prob = probability_table / np.sum(probability_table)

        # Sample
        logging.debug("Cond prob table: {}".format(condit_prob))
        sampled_state = np.random.multinomial(1, condit_prob)
        logging.debug("State sample: {}".format(sampled_state))

        z_k = np.argmax(sampled_state)
        logging.debug("Sampled Z_K: {}".format(z_k))
        z_list[k] = z_k
        return z_list

    def sample_states(self, observations_list, observed_sum):
        z_list = [-1 for x in observations_list]
        for i in range(len(z_list)):
            logging.info('Z_LIST is: {}'.format(z_list))
            z_list = self.sample_z_k(z_list, observations_list, observed_sum)

        return z_list
