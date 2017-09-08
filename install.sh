attach volume as /dev/sdf

sudo mount /dev/xvdf1 ramp
sudo chroot /home/ubuntu/ramp
exit
sudo mount --bind /dev /home/ubuntu/ramp
sudo chroot /home/ubuntu/ramp