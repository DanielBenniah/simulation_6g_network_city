import random
import heapq

class CommModule:
    """
    Simulates 6G V2V and V2I communication for vehicles and infrastructure.
    Supports direct and broadcast messages, network delay, and message drop.

    Parameters:
        delay_range (tuple): (min_delay, max_delay) in seconds for message delivery.
        drop_prob (float): Probability of message drop (0.0-1.0).
    Notes:
        - All times are in seconds.
        - Assumes ideal 6G network except for delay and drop probability.
        - Message recipients can be vehicle indices or string IDs (e.g., 'intersection').
    """
    def __init__(self, delay_range=(0.005, 0.05), drop_prob=0.001):
        """
        Initialize the CommModule.

        Args:
            delay_range (tuple): (min_delay, max_delay) in seconds.
            drop_prob (float): Probability of message drop.
        """
        self.delay_range = delay_range  # (min_delay, max_delay) in seconds
        self.drop_prob = drop_prob      # Probability of message drop
        self.queue = []  # Heap of (deliver_time, recipient, message)
        self.current_time = 0.0

    def send(self, message, recipient, now=None):
        """
        Send a direct message to a recipient (vehicle or intersection).
        Message will be delivered after a random delay, unless dropped.

        Args:
            message (dict): The message to send.
            recipient (int or str): Recipient vehicle index or intersection ID.
            now (float, optional): Current simulation time (seconds).
        """
        if random.random() < self.drop_prob:
            return  # Message dropped
        delay = random.uniform(*self.delay_range)
        deliver_time = (now if now is not None else self.current_time) + delay
        heapq.heappush(self.queue, (deliver_time, recipient, message))

    def broadcast(self, message, sender, recipients, now=None):
        """
        Broadcast a message from sender to a list of recipients (e.g., all vehicles in range).

        Args:
            message (dict): The message to broadcast.
            sender (int): The sender's vehicle index.
            recipients (iterable): List of recipient indices.
            now (float, optional): Current simulation time (seconds).
        """
        for recipient in recipients:
            if recipient == sender:
                continue
            self.send(message, recipient, now=now)

    def deliver_messages(self, current_time):
        """
        Deliver all messages whose deliver_time <= current_time.

        Args:
            current_time (float): Current simulation time (seconds).
        Returns:
            list: List of (recipient, message) tuples delivered at this time.
        """
        delivered = []
        self.current_time = current_time
        while self.queue and self.queue[0][0] <= current_time:
            _, recipient, message = heapq.heappop(self.queue)
            delivered.append((recipient, message))
        return delivered

    def clear(self):
        """
        Clear all pending messages from the queue.
        """
        self.queue.clear() 