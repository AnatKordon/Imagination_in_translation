from models.gpt_model import send_gpt_request
prompt = """
# A children's book drawing of a veterinarian using a stethoscope to 
# listen to the heartbeat of a baby otter.
# """
send_gpt_request(prompt)