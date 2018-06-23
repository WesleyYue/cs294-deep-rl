import unittest
from unittest.mock import MagicMock, Mock, patch

from agent import Agent
from multiprocessing import JoinableQueue
import time
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format=
    "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout)


class TestAgent(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    # Test that agent does not transition to UPDATE state from UDPATE state
    # (without first passing through TRAINING or ROLLOUT)
    def test_consecutive_updates_disallowed(self):
        """A finnicky and bad test to test that agent does not transition to 
        UPDATE state from UDPATE state. Relies on ASSERT inside UPDATE logic for
        the test.
        """
        mock_agent1 = Agent()
        mock_agent2 = Agent()

        state_queue = JoinableQueue()
        mock_agent1.state_queue = state_queue
        mock_agent2.state_queue = state_queue

        # Mock fake queues needed in UPDATE and TERMINATE states
        mock_joinable_queue = MagicMock(JoinableQueue())
        mock_agent1.network_weights = mock_joinable_queue
        mock_agent2.network_weights = mock_joinable_queue
        mock_agent1.results = mock_joinable_queue
        mock_agent2.results = mock_joinable_queue
        mock_agent1.paths_queue = mock_joinable_queue
        mock_agent2.paths_queue = mock_joinable_queue
        mock_agent1._load_weights = MagicMock(Agent._load_weights)
        mock_agent2._load_weights = MagicMock(Agent._load_weights)

        # Spawn agents
        mock_agent1.start()
        mock_agent2.start()

        # Put two UPDATE states in queue
        mock_agent1.state_queue.put(Agent.States.UPDATE)
        mock_agent1.state_queue.put(Agent.States.UPDATE)

        # Kill all processes
        mock_agent1.state_queue.put(Agent.States.TERMINATE)
        mock_agent1.state_queue.put(Agent.States.TERMINATE)

        mock_agent1.join()
        mock_agent2.join()

    # def test_something_else(self):
    # 	my_thing = MyClass()
    #     assertNotEqual(my_thing('a'), my_thing('b'))


if __name__ == '__main__':
    unittest.main()
