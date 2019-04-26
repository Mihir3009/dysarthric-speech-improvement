

# cd /media/mihir/Dysarthia/dysarthic_interspeech/UA/gender_specific/codes/ && python3 encoder.py encoder female
# echo "Encoder Female Done"

# cd /media/mihir/Dysarthia/dysarthic_interspeech/UA/gender_specific/codes/ && python3 encoder.py encoder male
# echo "Encoder Male Done"




cd /media/mihir/Dysarthia/dysarthic_interspeech/UA/gender_specific/codes/ && python3 DNN.py DNN female
echo "DNN Female Done"

cd /media/mihir/Dysarthia/dysarthic_interspeech/UA/gender_specific/codes/ && python3 DNN.py DNN male
echo "DNN Male Done"




cd /media/mihir/Dysarthia/dysarthic_interspeech/UA/gender_specific/codes/ && python3 gan_dnn.py GAN female
echo "GAN Female Done"


cd /media/mihir/Dysarthia/dysarthic_interspeech/UA/gender_specific/codes/ && python3 gan_dnn.py GAN male
echo "GAN Male Done"




cd /media/mihir/Dysarthia/dysarthic_interspeech/UA/gender_specific/codes/ && python3 DiscoGAN_torch.py DiscoGAN female
echo "DiscoGAN Female Done"


cd /media/mihir/Dysarthia/dysarthic_interspeech/UA/gender_specific/codes/ && python3 DiscoGAN_torch.py DiscoGAN male
echo "DiscoGAN Male Done"
