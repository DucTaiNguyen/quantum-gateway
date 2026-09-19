from enum import Enum


class NoiseAction(str, Enum):
    LEARN = "LEARN"
    INVESTIGATE = "INVESTIGATE"
    MITIGATE = "MITIGATE"
    IGNORE = "IGNORE"


def decide(
    utility,
    tv,
    learn_threshold=0.02,
    severe_noise_threshold=0.20,
):
    if utility >= learn_threshold:
        return NoiseAction.LEARN.value

    if tv >= severe_noise_threshold:
        return NoiseAction.MITIGATE.value

    if utility > 0:
        return NoiseAction.INVESTIGATE.value

    return NoiseAction.IGNORE.value
