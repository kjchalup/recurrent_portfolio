import boto3
import time
import os
ImageId = 'ami-16348b76'
KeyName = 'causeai'
InstanceType = 'p2.xlarge'
AvailabilityZone = 'us-west-2c'
SpotPrice = '0.90'
VolumeId='vol-6e1ab6fb'
VolumeId_test= 'vol-0e8c375d455ee5772'
DEVICE = '/dev/xvdf'
SSH_OPTS        = '-i ~/.aws/causeai.pem'
BLOCK_DURATION_MINUTES = 180
ec2=boto3.resource('ec2')

ec2Client = boto3.client('ec2')
req=ec2Client.request_spot_instances(DryRun=False,
                                     SpotPrice=SpotPrice,
                                     InstanceCount=1,
                                     Type='one-time',
                                     LaunchSpecification={'ImageId': ImageId,
                                                          'KeyName': KeyName,
                                                          'InstanceType': InstanceType,
                                                          'SecurityGroups': ['SSH_main','default'],
                                                          'Placement': {'AvailabilityZone': 
                                                                        AvailabilityZone,}})

req_id=req['SpotInstanceRequests'][0]['SpotInstanceRequestId']
print 'Waiting for instance {0} to switch to running state'.format(req_id)
time.sleep(5)
waiter = ec2Client.get_waiter('spot_instance_request_fulfilled')
waiter.wait(SpotInstanceRequestIds=[req_id])
req = ec2Client.describe_spot_instance_requests(SpotInstanceRequestIds=[req_id])
instance_id = req['SpotInstanceRequests'][0]['InstanceId']
waiter = ec2Client.get_waiter('instance_running')
waiter.wait(InstanceIds=[instance_id])
print 'Attaching volume {0} to device {1}'.format(VolumeId, DEVICE)
volume = ec2.Volume(VolumeId)
volume.attach_to_instance(InstanceId=instance_id,Device=DEVICE)
print 'Waiting for volume to switch to In Use state'
waiter = ec2Client.get_waiter('volume_in_use')
waiter.wait(VolumeIds=[VolumeId])
print 'Volume is attached'
print 'Waiting for the instance to finish booting'
time.sleep(60)
req=ec2Client.describe_instances(InstanceIds=[instance_id])
public_ip_address=req['Reservations'][0]['Instances'][0]['PublicIpAddress']
print "This is the IP address: " + public_ip_address
print 'Mounting the volume'
os.system("ssh {0} ubuntu@{1} \"sudo mount {2} data\"".format(SSH_OPTS, public_ip_address, DEVICE))
print "This is the IP address: " + public_ip_address
