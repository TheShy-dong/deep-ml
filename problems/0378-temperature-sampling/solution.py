import numpy as np

def temperature_sampling(logits: np.ndarray, temperature: float) -> list:
	"""
	Compute temperature-scaled softmax probabilities from logits.
	
	Args:
		logits: 1D numpy array of raw model output scores
		temperature: float controlling distribution sharpness
	
	Returns:
		List of probabilities after temperature scaling
	"""
	# Your code here
	if temperature <= 0:
        probs = np.zeros_like(logits, dtype=float)
        probs[np.argmax(logits)] = 1.0
        return probs.tolist()
	logits=logits/temperature
	logits=logits-np.max(logits)
	logits=np.exp(logits)
	return logits/np.sum(logits)
	pass