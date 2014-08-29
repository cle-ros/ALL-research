__author__ = 'Will Dabney, Clemens Rosenbaum'

# Original implementation by Will Dabney; adapted for working with decision diagrams by Clemens Rosenbaum

import copy

import numpy
from rlglue.types import Action

from pyrl.rlglue.registry import register_agent
from pyrl.agents import sarsa_lambda


@register_agent
class QlearningAgent(sarsa_lambda.sarsa_lambda):
    name = "Q-Learning"

    def agent_step(self, reward, observation):
        """Take one step in an episode for the agent, as the result of taking the last action.

        Args:
            reward: The reward received for taking the last action from the previous state.
            observation: The next observation of the episode, which is the consequence of taking the previous action.

        Returns:
            The next action the RL agent chooses to take, represented as an RLGlue Action object.
        """

        new_state = numpy.array(list(observation.doubleArray))
        last_state = numpy.array(list(self.last_observation.doubleArray))
        last_action = self.last_action.intArray[0]

        new_disc_state = self.getDiscState(observation.intArray)
        last_disc_state = self.getDiscState(self.last_observation.intArray)

        # Update eligibility traces
        phi_t = numpy.zeros(self.traces.shape)
        phi_t[last_disc_state, :, last_action] = self.basis.computeFeatures(last_state)

        self.update_traces(phi_t, None)
        self.update(phi_t, new_state, new_disc_state, reward)

        # QLearning can choose action after update
        new_int_action = self.getAction(new_state, new_disc_state)
        return_action = Action()
        return_action.intArray = [new_int_action]

        self.last_action = copy.deepcopy(return_action)
        self.last_observation = copy.deepcopy(observation)
        return return_action

    def get_action_values(self, state, discState):
        if state is not None:
            return numpy.dot(self.weights[discState, :, :].T, self.basis.computeFeatures(state))
        else:
            return numpy.zeros((self.numActions,))

    def update(self, phi_t, state, discState, reward):
        qvalues = self.get_action_values(state, discState)
        a_tp = qvalues.argmax()
        phi_tp = numpy.zeros(self.traces.shape)
        if state is not None:
            phi_tp[discState, :, a_tp] = self.basis.computeFeatures(state)

        # Compute Delta (TD-error)
        delta = self.gamma * qvalues[a_tp] + reward - numpy.dot(self.weights.flatten(), phi_t.flatten())

        # Update the weights with both a scalar and vector stepsize used
        # (Maybe we should actually make them both work together naturally)
        self.weights += self.rescale_update(phi_t, phi_tp, delta, reward, delta * self.traces)

    def agent_end(self, reward):
        """Receive the final reward in an episode, also signaling the end of the episode.

        Args:
            reward: The reward received for taking the last action from the previous state.
        """

        lastState = numpy.array(list(self.last_observation.doubleArray))
        lastAction = self.last_action.intArray[0]

        lastDiscState = self.getDiscState(self.last_observation.intArray)

        # Update eligibility traces
        phi_t = numpy.zeros(self.traces.shape)
        phi_t[lastDiscState, :, lastAction] = self.basis.computeFeatures(lastState)

        self.update_traces(phi_t, None)
        self.update(phi_t, None, 0, reward)


if __name__ == "__main__":
    from pyrl.agents.skeleton_agent import runAgent

    runAgent(QlearningAgent)





