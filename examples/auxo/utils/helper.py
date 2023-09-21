

def decode_msg(msg):
    """Decode message into event type and cohort id

    Args:
        msg (string): message from client
    """
    return msg.split('-')[0], int(msg.split('-')[1])


def generate_msg( msg_type, cohort_id=0):
    return f'{msg_type}-{cohort_id}'


