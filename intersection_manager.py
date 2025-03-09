import heapq

class IntersectionManager:
    """
    Manages reservations for a single intersection using a reservation-based system.
    Vehicles request to enter by providing approach parameters (arrival_time, speed, direction).
    The manager grants a reservation (time slot and path) if no conflict exists, otherwise asks the vehicle to slow/stop.
    Uses a first-come-first-served strategy.
    """
    def __init__(self):
        # Each reservation: (start_time, end_time, vehicle_id, path)
        self.reservations = []  # List of (start_time, end_time, vehicle_id, path)

    def request_reservation(self, vehicle_id, arrival_time, duration, path):
        """
        Request a reservation for a vehicle to pass through the intersection.
        Args:
            vehicle_id: Unique identifier for the vehicle.
            arrival_time: Proposed entry time to the intersection.
            duration: Time to cross the intersection.
            path: Path through the intersection (e.g., (from_dir, to_dir)).
        Returns:
            granted: True if reservation is granted, False otherwise.
            slot: (start_time, end_time) if granted, else suggested new time.
        """
        start_time = arrival_time
        end_time = arrival_time + duration
        # Check for conflicts
        for res in self.reservations:
            res_start, res_end, _, res_path = res
            if self.paths_conflict(path, res_path):
                # Overlapping time and conflicting path
                if not (end_time <= res_start or start_time >= res_end):
                    # Conflict detected
                    return False, res_end  # Suggest to try after res_end
        # No conflict, grant reservation
        heapq.heappush(self.reservations, (start_time, end_time, vehicle_id, path))
        return True, (start_time, end_time)

    def paths_conflict(self, path1, path2):
        """
        Returns True if two paths through the intersection would conflict (i.e., could collide).
        For simplicity, any crossing or same path is a conflict.
        path: (from_dir, to_dir), e.g., (0, 1) for North to East.
        """
        # If paths are the same or cross, consider conflict
        if path1 == path2:
            return True
        # For simplicity, all left turns, straight, and right turns can conflict if not orthogonal
        # (This can be made more sophisticated)
        if path1[0] == path2[0] or path1[1] == path2[1]:
            return True
        # Diagonal crossing (e.g., N->E and E->N) can also conflict
        if path1[0] == path2[1] and path1[1] == path2[0]:
            return True
        return False

    def cleanup(self, current_time):
        """
        Remove expired reservations (where end_time < current_time).
        """
        self.reservations = [res for res in self.reservations if res[1] > current_time]
        heapq.heapify(self.reservations)

    def get_reservations(self):
        """
        Return a list of current reservations for inspection or debugging.
        """
        return list(self.reservations) 