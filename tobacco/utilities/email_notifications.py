import boto.ses
from tobacco.secrets import get_secret

def send_email(subject, body):

    con = boto.ses.connect_to_region('us-west-2',
                                     aws_access_key_id=get_secret('tob_access'),
                                     aws_secret_access_key=get_secret('tob_access_sec'))

    con.send_email(get_secret('personal_email_address'), subject, body, [get_secret('personal_email_address')])


if __name__ == "__main__":
    send_email('test', 'test here')