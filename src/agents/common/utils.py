"""
Common utilities for all agents.
"""

def schedule(cfg_string, step):
    """Parse and evaluate a scheduling string.

    Supports linear scheduling in format: "linear(start,end,duration)"
    Example: "linear(1.0,0.1,50000)" will linearly decrease from 1.0 to 0.1 over 50000 steps

    Args:
        cfg_string (str): Configuration string specifying the schedule type and parameters
        step (int): Current step number

    Returns:
        float: Scheduled value for the current step
    """
    try:
        # Parse linear schedule
        if cfg_string.startswith('linear('):
            # Extract parameters from string
            params = cfg_string.strip('linear()').split(',')
            if len(params) != 3:
                raise ValueError("Linear schedule requires 3 parameters: start, end, duration")
            
            start = float(params[0])
            end = float(params[1])
            duration = int(params[2])

            # Compute linear interpolation
            if step >= duration:
                return end
            else:
                return start + (end - start) * (step / duration)

    except Exception as e:
        raise ValueError(f"Invalid schedule string format: {cfg_string}. Error: {str(e)}")

    raise ValueError(f"Unknown schedule type in string: {cfg_string}")