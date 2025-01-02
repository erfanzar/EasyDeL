import unittest
from easydel.trainers.auto_tx import get_optimizer_and_scheduler
from easydel.infra.etils import EasyDeLOptimizers, EasyDeLSchedulers

# FILE: easydel/trainers/test_auto_tx.py

import fjformer.optimizers


class TestGetOptimizerAndScheduler(unittest.TestCase):
	def setUp(self):
		self.steps = 1000
		self.learning_rate = 1e-5
		self.learning_rate_end = 1e-5
		self.gradient_accumulation_steps = 1
		self.weight_decay = 0.02
		self.warmup_steps = 0
		self.clip_grad = None
		self.mu_dtype = None

	def test_adafactor_with_linear_scheduler(self):
		optimizer, scheduler = get_optimizer_and_scheduler(
			EasyDeLOptimizers.ADAFACTOR,
			EasyDeLSchedulers.LINEAR,
			self.steps,
			self.learning_rate,
			self.learning_rate_end,
			self.gradient_accumulation_steps,
			self.weight_decay,
			self.warmup_steps,
			self.clip_grad,
			self.mu_dtype,
		)
		self.assertEqual(optimizer, fjformer.optimizers.get_adafactor_with_linear_scheduler)

	def test_lion_with_cosine_scheduler(self):
		optimizer, scheduler = get_optimizer_and_scheduler(
			EasyDeLOptimizers.LION,
			EasyDeLSchedulers.COSINE,
			self.steps,
			self.learning_rate,
			self.learning_rate_end,
			self.gradient_accumulation_steps,
			self.weight_decay,
			self.warmup_steps,
			self.clip_grad,
			self.mu_dtype,
		)
		self.assertEqual(optimizer, fjformer.optimizers.get_lion_with_cosine_scheduler)

	def test_adamw_with_warmup_cosine_scheduler(self):
		optimizer, scheduler = get_optimizer_and_scheduler(
			EasyDeLOptimizers.ADAMW,
			EasyDeLSchedulers.WARM_UP_COSINE,
			self.steps,
			self.learning_rate,
			self.learning_rate_end,
			self.gradient_accumulation_steps,
			self.weight_decay,
			self.warmup_steps,
			self.clip_grad,
			self.mu_dtype,
		)
		self.assertEqual(
			optimizer, fjformer.optimizers.get_adamw_with_warmup_cosine_scheduler
		)

	def test_rmsprop_with_warmup_linear_scheduler(self):
		optimizer, scheduler = get_optimizer_and_scheduler(
			EasyDeLOptimizers.RMSPROP,
			EasyDeLSchedulers.WARM_UP_LINEAR,
			self.steps,
			self.learning_rate,
			self.learning_rate_end,
			self.gradient_accumulation_steps,
			self.weight_decay,
			self.warmup_steps,
			self.clip_grad,
			self.mu_dtype,
		)
		self.assertEqual(
			optimizer, fjformer.optimizers.get_rmsprop_with_warmup_linear_scheduler
		)

	def test_invalid_optimizer(self):
		with self.assertRaises(ValueError):
			get_optimizer_and_scheduler(
				"INVALID_OPTIMIZER",
				EasyDeLSchedulers.LINEAR,
				self.steps,
				self.learning_rate,
				self.learning_rate_end,
				self.gradient_accumulation_steps,
				self.weight_decay,
				self.warmup_steps,
				self.clip_grad,
				self.mu_dtype,
			)

	def test_invalid_scheduler(self):
		with self.assertRaises(ValueError):
			get_optimizer_and_scheduler(
				EasyDeLOptimizers.ADAFACTOR,
				"INVALID_SCHEDULER",
				self.steps,
				self.learning_rate,
				self.learning_rate_end,
				self.gradient_accumulation_steps,
				self.weight_decay,
				self.warmup_steps,
				self.clip_grad,
				self.mu_dtype,
			)


if __name__ == "__main__":
	unittest.main()
