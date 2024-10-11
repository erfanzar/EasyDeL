import threading


class ReqIDGenerator:
	def __init__(self):
		self.current_id = 0
		self.lock = threading.Lock()

	def generate_id(self):
		with self.lock:
			id = self.current_id
			self.current_id += 8
		return id


def convert_sub_id_to_group_id(sub_req_id):
	return (sub_req_id // 8) * 8
